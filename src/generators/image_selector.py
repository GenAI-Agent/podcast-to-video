#!/usr/bin/env python3
"""
Image Selector Module
Handles image selection and timeline assignment for video generation
"""

import os
import sys
import json
import re
from typing import List, Dict, Optional, Tuple
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Add project root to sys.path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.database.pinecone_handler import PineconeHandler
from src.utils.json_image_loader import JsonImageLoader

# Load environment variables
load_dotenv()

# GPT Prompt for image descriptions
IMAGE_SCRIPT_PROMPT = """
You are a professional video director. Based on the provided transcript and audio duration, generate a script with image descriptions and durations.

Each image should:
- Have a clear, searchable description (tags/keywords that can be used to find relevant images)
- Have a duration of at least 1 second
- Together, all images should cover the entire audio duration
- Be relevant to the content being spoken at that time

The descriptions should be specific and visual, suitable for image search.
"""

IMAGE_SCRIPT_USER_PROMPT = """
Based on the following transcript and audio duration, generate a script with image descriptions and durations.

Transcript:
{transcript}

Audio Duration: {audio_duration} seconds

Generate a list of image descriptions with durations that cover the entire audio. Each item should have:
- description: specific visual keywords/tags for image search
- duration: time in seconds (minimum 1 second)

Return ONLY a JSON array in this exact format:
[
  {{"description": "business meeting, professionals discussing", "duration": 2.5}},
  {{"description": "stock market graphs, financial charts", "duration": 3.0}},
  ...
]

Ensure:
1. The sum of all durations equals the audio duration
2. Each duration is at least 1 second
3. Descriptions are specific and searchable
4. The number of images is appropriate for the content (not too many, not too few)
"""

class ImageSelector:
    def __init__(self, restricted_json_file: str = None, topic: str = None):
        """
        Initialize the image selector
        
        Args:
            restricted_json_file: Optional path to JSON file containing restricted image dataset
            topic: Selected topic for category-based filtering
        """
        self.pinecone_handler = PineconeHandler()
        self.topic = topic
        self.restricted_json_file = restricted_json_file
        self.json_image_loader = None
        
        # Initialize restricted dataset if provided
        if restricted_json_file and os.path.exists(restricted_json_file):
            self.json_image_loader = JsonImageLoader(restricted_json_file)
        
        # Topic categories mapping
        self.topic_categories = self._get_topic_categories()
        
        # Initialize Azure OpenAI for image script generation
        self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-testing")
        
        if self.api_base and self.api_key:
            self.llm = AzureChatOpenAI(
                azure_endpoint=self.api_base,
                api_key=self.api_key,
                api_version="2025-01-01-preview",
                azure_deployment=self.deployment_name,
                temperature=0.7,
                max_tokens=2000
            )
        else:
            self.llm = None
            print("Warning: Azure OpenAI not configured, image script generation will be limited")
    
    def _get_topic_categories(self) -> dict:
        """Get topic to category mappings"""
        return {
            'trading': ['business', 'finance', 'trading', 'market', 'investment'],
            'fantasy': ['fantasy', 'magic', 'adventure', 'mystical', 'epic'],
            'astrology': ['astrology', 'zodiac', 'celestial', 'cosmic', 'spiritual'],
            'romance': ['romance', 'love', 'couple', 'romantic', 'relationship'],
            'horror': ['horror', 'dark', 'scary', 'spooky', 'mysterious'],
            'drama': ['drama', 'emotional', 'intense', 'theatrical', 'story'],
            'thriller': ['thriller', 'suspense', 'action', 'tension', 'mystery']
        }
    
    def generate_image_timeline(self, transcript: str, audio_duration: float) -> List[Dict]:
        """
        Generate image timeline with descriptions and durations
        
        Args:
            transcript: Video script text
            audio_duration: Duration of audio in seconds
            
        Returns:
            List of dictionaries with image descriptions and durations
        """
        try:
            if self.llm:
                # Use GPT to generate structured image script
                prompt = ChatPromptTemplate.from_messages([
                    ("system", IMAGE_SCRIPT_PROMPT),
                    ("user", IMAGE_SCRIPT_USER_PROMPT)
                ])
                
                chain = prompt | self.llm | StrOutputParser()
                response = chain.invoke({
                    "transcript": transcript,
                    "audio_duration": audio_duration
                })
                
                # Parse JSON response
                try:
                    image_script = json.loads(response.strip())
                    
                    # Validate the generated script
                    if self._validate_image_script(image_script, audio_duration):
                        print(f"✓ Successfully generated {len(image_script)} image descriptions with durations")
                        return image_script
                    else:
                        print("⚠ Generated image script validation failed, using fallback")
                        
                except json.JSONDecodeError:
                    print("⚠ Failed to parse GPT response as JSON, using fallback")
            
            # Fallback: create basic timeline from sentences
            return self._create_fallback_timeline(transcript, audio_duration)
            
        except Exception as e:
            print(f"Error generating image timeline: {e}")
            return self._create_fallback_timeline(transcript, audio_duration)
    
    def _validate_image_script(self, image_script: List[Dict], target_duration: float) -> bool:
        """Validate that image script is properly formatted and timed"""
        if not isinstance(image_script, list) or len(image_script) == 0:
            return False
        
        total_duration = 0
        for item in image_script:
            if not isinstance(item, dict):
                return False
            if "description" not in item or "duration" not in item:
                return False
            if not isinstance(item["description"], str) or not item["description"].strip():
                return False
            if not isinstance(item["duration"], (int, float)) or item["duration"] < 1:
                return False
            total_duration += item["duration"]
        
        # Allow 10% tolerance in total duration
        return abs(total_duration - target_duration) <= target_duration * 0.1
    
    def _create_fallback_timeline(self, transcript: str, audio_duration: float) -> List[Dict]:
        """Create basic image timeline from transcript sentences"""
        # Split transcript into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
        
        if not sentences:
            return [{"description": "general image", "duration": audio_duration}]
        
        # Distribute duration evenly across sentences
        duration_per_sentence = audio_duration / len(sentences)
        
        timeline = []
        for sentence in sentences:
            # Use first few words as description
            words = sentence.split()[:8]  # First 8 words
            description = " ".join(words)
            
            timeline.append({
                "description": description,
                "duration": round(duration_per_sentence, 2)
            })
        
        print(f"✓ Created fallback timeline with {len(timeline)} segments")
        return timeline
    
    def search_images_for_timeline(self, timeline: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Search for images matching the timeline descriptions
        
        Args:
            timeline: List of image descriptions with durations
            top_k: Number of image candidates to return per description
            
        Returns:
            List of timeline items with assigned image paths and metadata
        """
        try:
            results = []
            used_images = set()
            
            # Show filtering status
            if self.topic and self.topic in self.topic_categories:
                print(f"🎯 Topic-based filtering active: '{self.topic}' -> categories {self.topic_categories[self.topic]}")
            elif self.json_image_loader:
                print(f"🔒 Restricted dataset mode (no topic filter)")
            else:
                print(f"🌍 Full dataset mode (no restrictions)")
            
            for i, item in enumerate(timeline):
                description = item["description"]
                duration = item["duration"]
                
                # Search for images
                image_candidates = self._search_images_for_description(description, top_k)
                
                # Select best available image
                selected_image = None
                for candidate in image_candidates:
                    if candidate["file_path"] not in used_images:
                        selected_image = candidate
                        used_images.add(candidate["file_path"])
                        break
                
                # If no unique image found, allow duplicates
                if not selected_image and image_candidates:
                    selected_image = image_candidates[0]
                
                # Create timeline item
                timeline_item = {
                    "index": i,
                    "description": description,
                    "duration": duration,
                    "image": selected_image,
                    "candidates": image_candidates[:5]  # Store top 5 alternatives
                }
                
                results.append(timeline_item)
                
                if selected_image:
                    print(f"Found image for segment {i+1}: '{description[:30]}...' -> {selected_image['file_path']}")
                else:
                    print(f"⚠ No image found for segment {i+1}: '{description[:30]}...'")
            
            found_count = len([r for r in results if r["image"]])
            print(f"📊 Image assignment complete: {found_count}/{len(timeline)} images assigned")
            
            return results
            
        except Exception as e:
            print(f"Error searching images for timeline: {e}")
            return []
    
    def _search_images_for_description(self, description: str, top_k: int = 10) -> List[Dict]:
        """Search for images matching a specific description"""
        try:
            # Use Pinecone to search for images
            results = self.pinecone_handler.query_pinecone(
                query=description,
                metadata_filter={},
                index_name="image-library",
                namespace="description",
                top_k=top_k
            )
            
            candidates = []
            for result in results:
                if "metadata" in result and "file_path" in result["metadata"]:
                    candidate_path = result["metadata"]["file_path"]
                    
                    # Check if image is allowed in restricted mode
                    if self.json_image_loader and not self._is_image_allowed(result):
                        continue
                    
                    # Validate that file exists
                    wsl_path = self._convert_windows_path_to_wsl(candidate_path)
                    if not os.path.exists(wsl_path):
                        continue
                    
                    # Create candidate info
                    candidate = {
                        "file_path": candidate_path,
                        "score": result.get("score", 0),
                        "metadata": result.get("metadata", {}),
                        "web_path": self._get_web_path(candidate_path)
                    }
                    
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            print(f"Error searching for description '{description}': {e}")
            return []
    
    def _is_image_allowed(self, search_result: dict) -> bool:
        """Check if image is allowed based on restrictions"""
        if not self.json_image_loader:
            return True
        
        try:
            file_path = search_result["metadata"]["file_path"]
            
            # Check if image exists in restricted dataset
            for img_info in self.json_image_loader.image_data:
                if img_info["file_path"] == file_path:
                    # Check topic categories if specified
                    if self.topic and self.topic in self.topic_categories:
                        img_tags = img_info.get("tags", "").lower()
                        topic_cats = [cat.lower() for cat in self.topic_categories[self.topic]]
                        return any(cat in img_tags for cat in topic_cats)
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _convert_windows_path_to_wsl(self, windows_path: str) -> str:
        """Convert Windows path to WSL path"""
        if windows_path.startswith("C:"):
            return windows_path.replace("C:", "/mnt/c").replace("\\", "/")
        return windows_path
    
    def _get_web_path(self, file_path: str) -> str:
        """Convert file path to web-accessible path"""
        wsl_path = self._convert_windows_path_to_wsl(file_path)
        
        # Create relative path for web serving
        allowed_bases = [
            "/home/fluxmind/batch_image/data",
            "/mnt/c/Users/x7048/Documents/ComfyUI/output"
        ]
        
        web_path = os.path.basename(wsl_path)  # fallback to filename
        for base_dir in allowed_bases:
            try:
                if wsl_path.startswith(base_dir):
                    web_path = os.path.relpath(wsl_path, base_dir)
                    break
            except ValueError:
                continue
        
        return web_path
    
    def replace_timeline_image(self, timeline: List[Dict], index: int, new_image_path: str = None) -> List[Dict]:
        """
        Replace an image in the timeline
        
        Args:
            timeline: Current timeline
            index: Index of item to replace
            new_image_path: Specific image path to use, or None to search for alternatives
            
        Returns:
            Updated timeline
        """
        if index < 0 or index >= len(timeline):
            raise ValueError("Invalid timeline index")
        
        timeline_item = timeline[index]
        
        if new_image_path:
            # Use specific image
            timeline_item["image"] = {
                "file_path": new_image_path,
                "score": 1.0,
                "metadata": {},
                "web_path": self._get_web_path(new_image_path)
            }
        else:
            # Search for alternative from candidates
            candidates = timeline_item.get("candidates", [])
            current_path = timeline_item["image"]["file_path"] if timeline_item["image"] else None
            
            # Find next best candidate
            for candidate in candidates:
                if candidate["file_path"] != current_path:
                    timeline_item["image"] = candidate
                    break
            else:
                # If no alternatives, search again
                new_candidates = self._search_images_for_description(
                    timeline_item["description"], top_k=20
                )
                used_paths = {item["image"]["file_path"] for item in timeline if item["image"]}
                
                for candidate in new_candidates:
                    if candidate["file_path"] not in used_paths:
                        timeline_item["image"] = candidate
                        timeline_item["candidates"] = new_candidates[:5]
                        break
        
        return timeline
    
    def get_timeline_preview(self, timeline: List[Dict]) -> List[Dict]:
        """
        Get preview information for timeline display
        
        Args:
            timeline: Image timeline
            
        Returns:
            List of preview information for frontend
        """
        previews = []
        
        for item in timeline:
            preview_info = {
                "index": item["index"],
                "description": item["description"][:100] + "..." if len(item["description"]) > 100 else item["description"],
                "duration": item["duration"],
                "has_image": item["image"] is not None,
                "image_path": item["image"]["web_path"] if item["image"] else None,
                "alternatives_count": len(item.get("candidates", [])),
                "error_reason": None if item["image"] else "No suitable image found"
            }
            
            previews.append(preview_info)
        
        return previews