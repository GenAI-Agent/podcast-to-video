#!/usr/bin/env python3
"""
JSON Image Dataset Loader
Utility class to load and parse batch image JSON files and extract available image information
"""

import json
import os
import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass


@dataclass
class ImageInfo:
    """Data class to hold image information from JSON dataset"""
    task_id: str
    prompt: str
    file_name: str
    category: str
    description: str = ""
    
    def __post_init__(self):
        """Extract description from prompt after initialization"""
        if self.prompt and not self.description:
            self.description = self._extract_description_from_prompt()
    
    def _extract_description_from_prompt(self) -> str:
        """Extract a meaningful description from the complex prompt JSON"""
        try:
            # Try to parse the prompt as JSON if it's a string
            if isinstance(self.prompt, str):
                # Look for JSON-like content in the prompt
                json_match = re.search(r'\{.*\}', self.prompt, re.DOTALL)
                if json_match:
                    prompt_json = json.loads(json_match.group())
                else:
                    return self.prompt[:100]  # Fallback to first 100 chars
            else:
                prompt_json = self.prompt
            
            # Extract description from various possible locations in the JSON
            if isinstance(prompt_json, dict):
                # Try different paths to find description
                paths_to_try = [
                    ['prompt', 'scene_description', 'title'],
                    ['prompt', 'description'],
                    ['prompt', 'scene_description', 'main_subject'],
                    ['prompt', 'scene_description', 'abstract_concept'],
                    ['prompt', 'scene_description'],
                    ['title'],
                    ['description'],
                    ['scene_description']
                ]
                
                for path in paths_to_try:
                    value = prompt_json
                    try:
                        for key in path:
                            if isinstance(value, dict) and key in value:
                                value = value[key]
                            else:
                                break
                        else:
                            # Successfully navigated the path
                            if isinstance(value, str) and value.strip():
                                return value.strip()
                    except (KeyError, TypeError):
                        continue
                
                # If no specific path worked, try to find any meaningful text
                return self._extract_any_meaningful_text(prompt_json)
            
            return str(prompt_json)[:100]
            
        except (json.JSONDecodeError, Exception) as e:
            # Fallback to raw prompt text
            return str(self.prompt)[:100] if self.prompt else ""
    
    def _extract_any_meaningful_text(self, data, max_length=100) -> str:
        """Recursively extract any meaningful text from nested JSON structure"""
        if isinstance(data, str):
            if len(data.strip()) > 10:  # Only return substantial text
                return data.strip()[:max_length]
        elif isinstance(data, dict):
            # Look for common description keys first
            priority_keys = ['title', 'description', 'concept', 'scene_description', 'main_subject']
            for key in priority_keys:
                if key in data:
                    result = self._extract_any_meaningful_text(data[key], max_length)
                    if result:
                        return result
            
            # Then check all other keys
            for key, value in data.items():
                if key not in priority_keys:
                    result = self._extract_any_meaningful_text(value, max_length)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = self._extract_any_meaningful_text(item, max_length)
                if result:
                    return result
        
        return ""


class JsonImageLoader:
    """Utility class to load and manage image datasets from JSON files"""
    
    def __init__(self, json_file_path: str):
        """
        Initialize the loader with a JSON file path
        
        Args:
            json_file_path: Path to the JSON file containing image data
        """
        self.json_file_path = json_file_path
        self.images: List[ImageInfo] = []
        self.images_by_category: Dict[str, List[ImageInfo]] = {}
        self.images_by_task_id: Dict[str, ImageInfo] = {}
        self.task_ids: Set[str] = set()
        
        self._load_data()
    
    def _load_data(self):
        """Load and parse the JSON data file"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            print(f"Loaded JSON data from: {self.json_file_path}")
            
            # Parse the data structure
            for category, items in data.items():
                if isinstance(items, list):
                    category_images = []
                    
                    for item in items:
                        if isinstance(item, dict) and 'task_id' in item:
                            image_info = ImageInfo(
                                task_id=item.get('task_id', ''),
                                prompt=item.get('prompt', ''),
                                file_name=item.get('file_name', ''),
                                category=category
                            )
                            
                            self.images.append(image_info)
                            category_images.append(image_info)
                            self.images_by_task_id[image_info.task_id] = image_info
                            self.task_ids.add(image_info.task_id)
                    
                    self.images_by_category[category] = category_images
            
            print(f"Loaded {len(self.images)} images across {len(self.images_by_category)} categories")
            for category, images in self.images_by_category.items():
                print(f"  - {category}: {len(images)} images")
                
        except FileNotFoundError:
            print(f"Error: File '{self.json_file_path}' not found.")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in file '{self.json_file_path}': {e}")
            raise
        except Exception as e:
            print(f"Error loading JSON data: {e}")
            raise
    
    def get_images_by_category(self, category: str) -> List[ImageInfo]:
        """Get all images for a specific category"""
        return self.images_by_category.get(category, [])
    
    def get_image_by_task_id(self, task_id: str) -> Optional[ImageInfo]:
        """Get a specific image by task ID"""
        return self.images_by_task_id.get(task_id)
    
    def get_all_task_ids(self) -> Set[str]:
        """Get all available task IDs"""
        return self.task_ids.copy()
    
    def get_all_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.images_by_category.keys())
    
    def get_descriptions_for_category(self, category: str) -> List[str]:
        """Get all descriptions for images in a specific category"""
        images = self.get_images_by_category(category)
        return [img.description for img in images if img.description]
    
    def search_images_by_keywords(self, keywords: List[str], category: str = None) -> List[ImageInfo]:
        """
        Search for images that match any of the given keywords
        
        Args:
            keywords: List of keywords to search for
            category: Optional category to limit search to
            
        Returns:
            List of matching ImageInfo objects
        """
        search_pool = self.images
        if category:
            search_pool = self.get_images_by_category(category)
        
        matches = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        for image in search_pool:
            # Search in description and prompt
            search_text = f"{image.description} {image.prompt}".lower()
            
            for keyword in keywords_lower:
                if keyword in search_text:
                    matches.append(image)
                    break  # Don't add the same image multiple times
        
        return matches
    
    def print_summary(self):
        """Print a summary of the loaded dataset"""
        print(f"\n=== JSON Image Dataset Summary ===")
        print(f"File: {self.json_file_path}")
        print(f"Total Images: {len(self.images)}")
        print(f"Categories: {len(self.images_by_category)}")
        
        for category, images in self.images_by_category.items():
            print(f"\n{category} ({len(images)} images):")
            for i, img in enumerate(images[:3], 1):  # Show first 3 as examples
                print(f"  {i}. {img.description[:60]}...")
            if len(images) > 3:
                print(f"  ... and {len(images) - 3} more")
