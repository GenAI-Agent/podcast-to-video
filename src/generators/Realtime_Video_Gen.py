#!/usr/bin/env python3
"""
Real-time Video Generator Script
Complete pipeline: Article → Script → Image Generation (ComfyUI) → Video
"""

import os
import sys
import json
import time
import tempfile
import librosa
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

# Import image and video generation modules
from .image_generator import generate_image_prompt_fun, call_image_request_function
from .video_generator import VideoGenerator

# Load environment variables
load_dotenv()

# Dual prompt system: First select art style, then generate image prompt

# PROMPT 1: Art Style Selection
ART_STYLE_PROMPT = """
[PLACEHOLDER FOR ART STYLE SELECTION PROMPT]

You are an art director who selects the perfect visual style for content. Based on the provided transcript, choose an appropriate art style that would best represent the content visually. If it's a children's book choose from the children's collection.

Transcript:
{transcript}

If it's a children's book choose one of these styles and additionaly enphises the use of vivid colours and cuteness:
Watercolour Painting
Pastel Crayon / Chalk
Coloured Pencil Sketch
Classic Cartoon Illustration
Exaggerated Caricature
Simple Line Art with Flat Colours
Fairy Tale Ink & Wash
Retro Golden Book Style
Cut-Paper Collage
Anime / Chibi Style
Digital Soft-Shaded
Vector Art (Flat & Clean)
Clay / Stop-Motion Look
Felt & Fabric Illustration
Woodblock Print-Inspired
Monochrome Line and Wash

If it's educational content choose one of these styles:
Flat Vector Design
Minimalist Line Art
Diagrammatic Illustration
Isometric Illustration
Blueprint / Technical Drawing Style
Monochrome Line and Wash
Photorealistic Illustration
Digital 3D Rendering
Data Visualisation Style
Pictogram / Icon-Based Design

If a novel, choose one of these styles:
Painterly Watercolour
Oil Painting Style
Gouache Illustration
Impressionist Brushwork
Digital Painting (Rich Colour)
Coloured Ink Wash
Art Nouveau Style
Fantasy Concept Art
Pastel Illustration
Surrealist Painting


Return ONLY the art style name. Based on the transcript, set the tone and vibe.
"""

# PROMPT 2: Image Prompt Generation with Style
IMAGE_PROMPT = """
[PLACEHOLDER FOR IMAGE PROMPT GENERATION]

You are a professional visual content creator. Using the provided transcript and the selected art style, generate a detailed image prompt that would be perfect for creating a visual representation of the spoken content.

Transcript:
{transcript}

Art Style: {art_style}

Generate a single, detailed image prompt that captures the essence of this content in the specified art style. The prompt should be:
- Descriptive and specific
- Incorporate the specified art style
- Visually compelling
- Relevant to the transcript content
- Suitable for AI image generation tools

Make sure to weave the art style naturally into the prompt description.
Return ONLY the image prompt text, nothing else.
"""

# PROMPT 3: Scene Breakdown for Video
SCENE_BREAKDOWN_PROMPT = """
You are a video director. Break down this script into distinct visual scenes for video creation.
Each scene should be 3-5 seconds long and have a clear visual description.

Script:
{script}

Total Duration: {duration} seconds

Create a list of scenes that cover the entire duration. Each scene should have:
- A clear visual description suitable for image generation
- Appropriate duration in seconds

Return ONLY a JSON array in this format:
[
  {{"description": "detailed visual description of scene 1", "duration": 3.5}},
  {{"description": "detailed visual description of scene 2", "duration": 4.0}},
  ...
]

Make sure the sum of all durations equals approximately {duration} seconds.
"""


class RealtimeVideoGenerator:
    def __init__(self, comfyui_url: str = None):
        """Initialize the real-time video generator with full pipeline support
        
        Args:
            comfyui_url: Optional ComfyUI API endpoint URL
        """
        self.temp_dir = tempfile.mkdtemp()
        self.comfyui_url = comfyui_url or os.getenv("COMFYUI_URL", "https://7fd6781ec07e.ngrok-free.app/api/prompt")
        
        # Initialize Azure OpenAI
        self.llm = None
        self._init_openai()
        
        # Initialize video generator for compilation
        self.video_generator = VideoGenerator()
        
        # Real-time generation state
        self.selected_art_style = None  # Store once-selected art style
        self.image_prompts = []  # Store all generated image prompts for ComfyUI
        self.image_timestamps = []  # Store timestamps for each image
        
        print(f"✓ Initialized with ComfyUI URL: {self.comfyui_url}")
    
    def _init_openai(self):
        """Initialize Azure OpenAI client"""
        try:
            api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            
            if not api_base or not api_key:
                print("⚠️ Warning: Azure OpenAI credentials not found")
                return
            
            self.llm = AzureChatOpenAI(
                azure_endpoint=api_base,
                api_key=api_key,
                azure_deployment="gpt-4o-testing",
                api_version="2025-01-01-preview",
                temperature=0.7,
                max_tokens=500,
                timeout=None,
                max_retries=2,
            )
            print("✓ Azure OpenAI initialized")
        except Exception as e:
            print(f"Error initializing OpenAI: {e}")
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcript text
        """
        # TODO: Implement audio transcription (e.g., Whisper API, speech recognition)
        # For now, return a placeholder
        print(f"📝 Transcribing audio: {audio_path}")
        
        # Placeholder transcript - replace with actual transcription
        return "This is a sample transcript generated from the audio file. In a real implementation, this would be the actual transcribed text from the audio."
    
    def select_art_style(self, transcript: str, custom_style_prompt: str = None) -> str:
        """
        Select appropriate art style based on transcript content
        
        Args:
            transcript: Audio transcript text
            custom_style_prompt: Optional custom prompt for style selection
            
        Returns:
            Selected art style
        """
        if not self.llm:
            print("❌ OpenAI not initialized")
            return "Photorealistic"  # Default fallback
        
        try:
            # Use custom prompt if provided, otherwise use default
            prompt = custom_style_prompt if custom_style_prompt else ART_STYLE_PROMPT
            
            # Format the prompt with transcript
            formatted_prompt = prompt.format(transcript=transcript)
            
            response = self.llm.invoke(formatted_prompt)
            art_style = response.content.strip()
            
            print(f"🎨 Selected art style: {art_style}")
            return art_style
            
        except Exception as e:
            print(f"Error selecting art style: {e}")
            return "Photorealistic"  # Default fallback
    
    def generate_image_prompt(self, transcript: str, art_style: str, custom_prompt: str = None) -> str:
        """
        Generate image prompt based on transcript and art style
        
        Args:
            transcript: Audio transcript text
            art_style: Selected art style
            custom_prompt: Optional custom prompt for image generation
            
        Returns:
            Generated image prompt
        """
        if not self.llm:
            print("❌ OpenAI not initialized")
            return "Error: OpenAI not available"
        
        try:
            # Use custom prompt if provided, otherwise use default
            prompt = custom_prompt if custom_prompt else IMAGE_PROMPT
            
            # Format the prompt with transcript and art style
            formatted_prompt = prompt.format(transcript=transcript, art_style=art_style)
            
            response = self.llm.invoke(formatted_prompt)
            image_prompt = response.content.strip()
            
            print("✓ Generated image prompt")
            return image_prompt
            
        except Exception as e:
            print(f"Error generating image prompt: {e}")
            return f"Error: {str(e)}"
    
    def process_audio_to_image_prompt(self, audio_path: str, custom_style_prompt: str = None, custom_image_prompt: str = None) -> Dict:
        """
        Main workflow: Convert audio to transcript, select art style, then generate image prompt
        
        Args:
            audio_path: Path to audio file
            custom_style_prompt: Optional custom system prompt for art style selection
            custom_image_prompt: Optional custom system prompt for image generation
            
        Returns:
            Dictionary with transcript, art style, and image prompt
        """
        print(f"🎵 Processing audio: {audio_path}")
        
        # Step 1: Transcribe audio
        transcript = self.transcribe_audio(audio_path)
        print(f"📝 Transcript: {transcript[:100]}...")
        
        # Step 2: Select art style
        art_style = self.select_art_style(transcript, custom_style_prompt)
        
        # Step 3: Generate image prompt with selected style
        image_prompt = self.generate_image_prompt(transcript, art_style, custom_image_prompt)
        print(f"🎨 Image prompt: {image_prompt[:100]}...")
        
        return {
            "transcript": transcript,
            "art_style": art_style,
            "image_prompt": image_prompt,
            "audio_path": audio_path
        }
    
    def process_text_to_image_prompt(self, text: str, custom_style_prompt: str = None, custom_image_prompt: str = None) -> Dict:
        """
        Generate image prompt directly from text using dual prompt system
        
        Args:
            text: Input text to use as transcript
            custom_style_prompt: Optional custom system prompt for art style selection
            custom_image_prompt: Optional custom system prompt for image generation
            
        Returns:
            Dictionary with transcript, art style, and image prompt
        """
        print(f"📝 Processing text: {text[:50]}...")
        
        # Step 1: Select art style
        art_style = self.select_art_style(text, custom_style_prompt)
        
        # Step 2: Generate image prompt with selected style
        image_prompt = self.generate_image_prompt(text, art_style, custom_image_prompt)
        print(f"🎨 Image prompt: {image_prompt[:100]}...")
        
        return {
            "transcript": text,
            "art_style": art_style,
            "image_prompt": image_prompt,
            "source": "text_input"
        }
    
    def break_script_into_scenes(self, script: str, duration: int = 60) -> List[Dict]:
        """
        Break script into visual scenes with timing for video
        
        Args:
            script: Video script text
            duration: Total duration in seconds
            
        Returns:
            List of scene dictionaries with descriptions and durations
        """
        if not self.llm:
            # Fallback: create basic scenes
            num_scenes = max(3, duration // 5)
            scene_duration = duration / num_scenes
            return [
                {"description": f"Scene {i+1} from script", "duration": scene_duration}
                for i in range(num_scenes)
            ]
        
        try:
            prompt = SCENE_BREAKDOWN_PROMPT.format(
                script=script,
                duration=duration
            )
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Parse JSON response
            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                scenes = json.loads(json_str)
                
                print(f"✓ Created {len(scenes)} scenes for video")
                return scenes
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            print(f"Error breaking into scenes: {e}")
            # Fallback
            num_scenes = max(3, duration // 5)
            scene_duration = duration / num_scenes
            return [
                {"description": f"Scene {i+1}", "duration": scene_duration}
                for i in range(num_scenes)
            ]
    
    def check_comfyui_queue_status(self, prompt_id: str) -> Dict:
        """
        Check ComfyUI queue status for a prompt
        
        Args:
            prompt_id: The ComfyUI prompt ID to check
            
        Returns:
            Dictionary with queue status information
        """
        try:
            import requests
            base_url = self.comfyui_url.replace("/api/prompt", "")
            queue_url = f"{base_url}/queue"
            
            response = requests.get(queue_url, timeout=10)
            if response.status_code == 200:
                queue_data = response.json()
                
                # Check if prompt is in running queue
                running = queue_data.get('queue_running', [])
                pending = queue_data.get('queue_pending', [])
                
                for item in running:
                    if item[1] == prompt_id:
                        return {"status": "running", "position": 0}
                
                for idx, item in enumerate(pending):
                    if item[1] == prompt_id:
                        return {"status": "pending", "position": idx + 1}
                
                return {"status": "completed_or_not_found", "position": -1}
            
            return {"status": "error", "position": -1}
            
        except Exception as e:
            print(f"Error checking queue status: {e}")
            return {"status": "error", "position": -1}
    
    def check_comfyui_completion_heartbeat(self, prompt_id: str, output_folder: str, max_attempts: int = 30) -> bool:
        """
        Heartbeat-based completion detection combining API polling with file existence checks
        
        Args:
            prompt_id: The ComfyUI prompt ID to check
            output_folder: Expected output folder name
            max_attempts: Maximum number of attempts to check
            
        Returns:
            True if completed and files exist, False otherwise
        """
        try:
            base_url = self.comfyui_url.replace("/api/prompt", "")
            history_url = f"{base_url}/history/{prompt_id}"
            
            print(f"💓 Starting heartbeat detection for prompt: {prompt_id}")
            print(f"📁 Monitoring folder: {output_folder}")
            
            # Define possible output paths for the expected files
            output_base_paths = [
                f"/mnt/c/Users/x7048/Documents/ComfyUI/output/{output_folder}",
                f"/mnt/c/Users/x7048/Documents/ComfyUI/output",
                "/mnt/c/Users/x7048/Documents/ComfyUI/output"
            ]
            
            for attempt in range(max_attempts):
                try:
                    import requests
                    
                    print(f"💓 Heartbeat {attempt + 1}/{max_attempts}")
                    
                    # Dual check: API status AND file existence
                    api_completed = False
                    files_found = []
                    
                    # 1. Check API status
                    queue_status = self.check_comfyui_queue_status(prompt_id)
                    
                    if queue_status["status"] == "pending":
                        print(f"📋 API: Prompt in queue, position: {queue_status['position']}")
                    elif queue_status["status"] == "running":
                        print(f"🔄 API: Prompt currently processing...")
                    elif queue_status["status"] == "completed_or_not_found":
                        # Check history for completion
                        response = requests.get(history_url, timeout=10)
                        
                        if response.status_code == 200:
                            history_data = response.json()
                            prompt_data = history_data.get(prompt_id, {})
                            
                            # Check if outputs exist (indicating completion)
                            outputs = prompt_data.get('outputs', {})
                            if outputs:
                                print(f"✅ API: Generation completed")
                                api_completed = True
                    
                    # 2. Check file existence (regardless of API status)
                    for base_path in output_base_paths:
                        if os.path.exists(base_path):
                            try:
                                # Look for recently created image/video files
                                for file in os.listdir(base_path):
                                    if any(ext in file.lower() for ext in ['.png', '.jpg', '.jpeg', '.mp4', '.mov', '.avi']):
                                        full_path = os.path.join(base_path, file)
                                        # Check if file was created recently (within last 10 minutes)
                                        file_age = time.time() - os.path.getctime(full_path)
                                        if file_age < 600:  # 10 minutes
                                            files_found.append(full_path)
                                            print(f"📁 FILE: Found recent file: {file}")
                            except Exception as e:
                                continue  # Skip this directory if there's an error
                    
                    # 3. Determine completion based on combined criteria
                    if files_found:
                        if api_completed:
                            print(f"✅ HEARTBEAT SUCCESS: Both API and files confirm completion!")
                            print(f"📁 Found {len(files_found)} generated files")
                            return True
                        else:
                            print(f"📁 FILES READY: Found {len(files_found)} files, but API still processing...")
                            # If files exist but API says still processing, wait a bit more
                            # This handles cases where files are ready but API hasn't updated
                            if attempt > max_attempts * 0.7:  # After 70% of attempts, trust files
                                print(f"✅ HEARTBEAT SUCCESS: Files exist, trusting file system over API")
                                return True
                    elif api_completed:
                        print(f"⚠️ API says completed but no files found yet, waiting...")
                    
                    if attempt < max_attempts - 1:
                        print(f"⏳ Waiting 3 seconds before next heartbeat...")
                    
                    time.sleep(3)  # Wait 3 seconds between checks
                    
                except Exception as e:
                    print(f"💓 Heartbeat error (attempt {attempt + 1}): {e}")
                    time.sleep(3)
            
            print(f"💔 Heartbeat timeout: No completion detected after {max_attempts} attempts")
            print(f"📊 Final status - API completed: {api_completed}, Files found: {len(files_found) if 'files_found' in locals() else 0}")
            return False
            
        except Exception as e:
            print(f"💔 Heartbeat system error: {e}")
            return False

    def check_comfyui_completion(self, prompt_id: str, max_attempts: int = 30) -> bool:
        """
        Legacy method - now redirects to heartbeat detection
        
        Args:
            prompt_id: The ComfyUI prompt ID to check
            max_attempts: Maximum number of attempts to check
            
        Returns:
            True if completed, False otherwise
        """
        # Extract output folder from current context or use timestamp
        output_folder = f"video_gen_{int(time.time())}"
        return self.check_comfyui_completion_heartbeat(prompt_id, output_folder, max_attempts)
    
    def get_generated_files(self, prompt_id: str, output_folder: str) -> List[str]:
        """
        Get list of generated files from ComfyUI output
        
        Args:
            prompt_id: The ComfyUI prompt ID
            output_folder: Output folder name
            
        Returns:
            List of file paths
        """
        try:
            # Common output patterns for ComfyUI
            output_base = "/mnt/c/Users/x7048/Documents/ComfyUI/output"
            possible_paths = [
                f"{output_base}/{output_folder}/",
                f"{output_base}/",
                f"{output_base}/generated/",
                f"{output_base}/images/",
                f"{output_base}/videos/"
            ]
            
            generated_files = []
            
            for base_path in possible_paths:
                if os.path.exists(base_path):
                    # Look for files that might be related to our prompt
                    for file in os.listdir(base_path):
                        if any(ext in file.lower() for ext in ['.png', '.jpg', '.jpeg', '.mp4', '.mov', '.avi']):
                            full_path = os.path.join(base_path, file)
                            # Check if file was created recently (within last 5 minutes)
                            if os.path.getctime(full_path) > time.time() - 300:
                                generated_files.append(full_path)
            
            # Sort by creation time, newest first
            generated_files.sort(key=lambda x: os.path.getctime(x), reverse=True)
            
            if generated_files:
                print(f"📁 Found {len(generated_files)} recently generated files")
                for file in generated_files[:3]:  # Show first 3
                    print(f"   - {file}")
            
            return generated_files
            
        except Exception as e:
            print(f"Error getting generated files: {e}")
            return []
    
    def generate_video_with_comfyui(self, transcript: str, art_style: str, duration: int) -> str:
        """
        Generate a complete video using ComfyUI with proper completion detection
        
        Args:
            transcript: The complete transcript text
            art_style: Selected art style
            duration: Target video duration in seconds
            
        Returns:
            Path to generated video file
        """
        print(f"🎬 Generating video with ComfyUI using IMAGE_PROMPT system...")
        
        try:
            # Generate comprehensive image prompt for the entire video
            image_prompt = self.generate_image_prompt(transcript, art_style)
            print(f"📝 Generated video prompt: {image_prompt[:100]}...")
            
            # Use the image prompt to generate a video via ComfyUI
            output_folder = f"video_gen_{int(time.time())}"
            
            # Submit video generation request
            video_prompt_id = self.call_video_generation_comfyui(image_prompt, output_folder, duration)
            
            if video_prompt_id:
                print(f"✓ Video generation submitted to ComfyUI: {video_prompt_id}")
                
                # Check for completion with heartbeat detection
                max_wait_attempts = max(20, duration * 2)  # Scale with video duration
                if self.check_comfyui_completion_heartbeat(video_prompt_id, output_folder, max_wait_attempts):
                    
                    # Look for generated files
                    generated_files = self.get_generated_files(video_prompt_id, output_folder)
                    
                    # Look for video files first
                    video_files = [f for f in generated_files if any(ext in f.lower() for ext in ['.mp4', '.mov', '.avi'])]
                    
                    if video_files:
                        video_path = video_files[0]  # Use the newest video file
                        print(f"✅ Video generated successfully: {video_path}")
                        return video_path
                    else:
                        print(f"⚠️ ComfyUI completed but no video file found")
                        return None
                else:
                    print(f"⚠️ ComfyUI video generation timed out")
                    return None
            else:
                print(f"❌ Failed to submit video generation to ComfyUI")
                return None
                
        except Exception as e:
            print(f"Error generating video with ComfyUI: {e}")
            return None
    
    def call_video_generation_comfyui(self, prompt: str, output_folder: str, duration: int) -> Optional[str]:
        """
        Call ComfyUI specifically for video generation
        
        Args:
            prompt: The comprehensive image prompt for the video
            output_folder: Output folder name
            duration: Video duration in seconds
            
        Returns:
            prompt_id if successful, None otherwise
        """
        try:
            # This should use a video generation workflow in ComfyUI
            # For now, we'll enhance the existing image workflow for video
            enhanced_prompt = f"Video sequence: {prompt}, duration: {duration} seconds, smooth transitions, cinematic"
            
            # Call ComfyUI with video-specific parameters
            prompt_id = call_image_request_function(enhanced_prompt, output_folder)
            
            return prompt_id
            
        except Exception as e:
            print(f"Error calling ComfyUI for video generation: {e}")
            return None
    
    def generate_single_image_with_comfyui(self, transcript: str, art_style: str) -> Optional[str]:
        """
        Fallback: Generate a single comprehensive image using ComfyUI with heartbeat detection
        
        Args:
            transcript: The complete transcript text
            art_style: Art style to apply
            
        Returns:
            Single generated image path or None
        """
        print(f"🖼️ Fallback: Generating single comprehensive image with ComfyUI...")
        
        try:
            output_folder = f"video_gen_{int(time.time())}"
            
            # Generate a comprehensive prompt for the entire content
            image_prompt = self.generate_image_prompt(transcript, art_style)
            print(f"📝 Generated comprehensive image prompt: {image_prompt[:100]}...")
            
            # Generate enhanced prompt using GPT
            enhanced_prompt = generate_image_prompt_fun(f"{image_prompt}, {art_style} style")
            
            print(f"🎨 Submitting single comprehensive image to ComfyUI...")
            
            # Call ComfyUI once for the entire content
            prompt_id = call_image_request_function(enhanced_prompt, output_folder)
            
            if prompt_id:
                print(f"✓ Submitted to ComfyUI: {prompt_id}")
                
                # Check completion using heartbeat detection
                if self.check_comfyui_completion_heartbeat(prompt_id, output_folder, max_attempts=20):
                    
                    # Look for generated files
                    generated_files = self.get_generated_files(prompt_id, output_folder)
                    
                    # Look for image files
                    image_files = [f for f in generated_files if any(ext in f.lower() for ext in ['.png', '.jpg', '.jpeg'])]
                    
                    if image_files:
                        image_path = image_files[0]  # Use the newest image file
                        print(f"✅ Single comprehensive image ready: {image_path}")
                        return image_path
                    else:
                        print(f"⚠️ ComfyUI completed but no image file found")
                        return None
                else:
                    print(f"💔 Heartbeat detection timed out")
                    return None
            else:
                print(f"❌ Failed to submit to ComfyUI")
                return None
                
        except Exception as e:
            print(f"Error generating single image with ComfyUI: {e}")
            return None
    
    def compile_video(self, image_paths: List[str], scenes: List[Dict], script: str, output_path: str) -> str:
        """
        Compile images into final video with audio and subtitles
        
        Args:
            image_paths: List of image file paths
            scenes: List of scene dictionaries with durations
            script: Video script for audio generation
            output_path: Output video file path
            
        Returns:
            Path to generated video
        """
        try:
            print("🎬 Compiling final video...")
            
            # Generate audio from script
            audio_path, audio_duration = self.video_generator.generate_audio(
                text_for_generation=script
            )
            print(f"🔊 Generated audio: {audio_duration:.2f}s")
            
            # Generate subtitles
            sentences = self.video_generator.split_article_into_sentences(script)
            srt_path = self.video_generator.generate_srt_file(sentences, audio_duration)
            print(f"📝 Generated subtitles")
            
            # Filter out None paths and use fallback if needed
            valid_paths = []
            for path in image_paths:
                if path and os.path.exists(path):
                    valid_paths.append(path)
                else:
                    # Use video_generator's search as fallback
                    print(f"  ⚠️ Missing image, using fallback search")
                    fallback = self.video_generator.vector_search_images([scenes[len(valid_paths)]["description"]])
                    if fallback and fallback[0]:
                        valid_paths.append(fallback[0])
            
            if not valid_paths:
                raise ValueError("No valid images for video compilation")
            
            # Create video with custom timing
            video_path = self.video_generator.create_video_with_image_script(
                image_paths=valid_paths,
                image_script=scenes[:len(valid_paths)],
                audio_path=audio_path,
                srt_path=srt_path,
                output_path=output_path,
                audio_duration=audio_duration,
                aspect_ratio="9:16"
            )
            
            print(f"✅ Video created: {video_path}")
            return video_path
            
        except Exception as e:
            print(f"Error compiling video: {e}")
            raise
    
    def process_article_to_video(
        self,
        article: str,
        output_path: str = "output_video.mp4",
        target_duration: int = 60,
        use_comfyui: bool = True,
        aspect_ratio: str = "9:16"
    ) -> str:
        """
        Complete pipeline: Article → Script → ComfyUI Video Generation OR Image Compilation
        
        Args:
            article: Input article text
            output_path: Output video file path
            target_duration: Target video duration in seconds
            use_comfyui: Whether to use ComfyUI for video generation
            aspect_ratio: Video aspect ratio ("9:16" or "16:9")
            
        Returns:
            Path to generated video
        """
        print("🚀 Starting article to video pipeline...")
        print(f"📄 Article length: {len(article)} characters")
        print(f"⏱️ Target duration: {target_duration} seconds")
        
        try:
            # Step 1: Process article to get art style
            result = self.process_text_to_image_prompt(article)
            art_style = result["art_style"]
            print(f"🎨 Selected art style: {art_style}")
            
            # Step 2: Generate single image with ComfyUI (only one call)
            if use_comfyui:
                print("🖼️ Generating single comprehensive image with ComfyUI...")
                single_image = self.generate_single_image_with_comfyui(article, art_style)
                
                if single_image:
                    print(f"✅ ComfyUI generated single image: {single_image}")
                else:
                    print("⚠️ ComfyUI image generation failed, using database fallback...")
                    use_comfyui = False  # Fall back to database images
            
            # Step 3: Break into scenes for video compilation
            print("📋 Breaking article into scenes...")
            scenes = self.break_script_into_scenes(article, target_duration)
            
            # Step 4: Prepare images based on generation method
            if use_comfyui and single_image:
                # Use the same single image for all scenes
                image_paths = [single_image] * len(scenes)
                print(f"✅ Using single ComfyUI image for all {len(scenes)} scenes")
            else:
                # Use existing database images
                print("🔍 Using existing images from database...")
                descriptions = [scene["description"] for scene in scenes]
                image_paths = self.video_generator.vector_search_images(descriptions)
            
            # Step 5: Compile video from images
            video_path = self.compile_video(
                image_paths=image_paths,
                scenes=scenes,
                script=article,
                output_path=output_path
            )
            
            print(f"\n✅ Pipeline complete! Video saved to: {video_path}")
            return video_path
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise
    
    def generate_realtime_images(self, user_input: str, video_duration: int = 60, images_per_second: float = 0.5) -> List[Dict]:
        """
        Generate images for video in real-time based on user input.
        Prompt 1 runs once to select art style, Prompt 2 runs for each image needed.
        
        Args:
            user_input: User's input text/transcript
            video_duration: Total video duration in seconds
            images_per_second: How many images per second (default: 1 image per 2 seconds)
            
        Returns:
            List of dictionaries with image prompts and timestamps
        """
        print(f"🎬 Starting real-time image generation for {video_duration}s video...")
        
        # Step 1: Select art style ONCE (only if not already selected)
        if not self.selected_art_style:
            print("🎨 Running Prompt 1: Art Style Selection...")
            self.selected_art_style = self.select_art_style(user_input)
            print(f"✓ Art style selected: {self.selected_art_style}")
        else:
            print(f"🎨 Using previously selected art style: {self.selected_art_style}")
        
        # Step 2: Calculate number of images needed
        total_images = max(1, int(video_duration * images_per_second))
        interval = video_duration / total_images
        
        print(f"📊 Generating {total_images} images (1 every {interval:.1f}s)")
        
        # Step 3: Generate image prompts with timestamps
        image_data = []
        self.image_prompts.clear()
        self.image_timestamps.clear()
        
        for i in range(total_images):
            timestamp = i * interval
            
            # Create context for this specific moment in the video
            progress_context = f"Scene {i+1} of {total_images} at {timestamp:.1f}s: {user_input}"
            
            print(f"🖼️ Running Prompt 2 for image {i+1}/{total_images} at {timestamp:.1f}s...")
            
            # Generate image prompt for this specific moment
            image_prompt = self.generate_image_prompt(progress_context, self.selected_art_style)
            
            # Store the prompt and timestamp
            self.image_prompts.append(image_prompt)
            self.image_timestamps.append(timestamp)
            
            image_data.append({
                "prompt": image_prompt,
                "timestamp": timestamp,
                "duration": interval,
                "index": i
            })
            
            print(f"✓ Generated prompt {i+1}: {image_prompt[:60]}...")
        
        print(f"✅ Generated {len(image_data)} image prompts for ComfyUI")
        return image_data
    
    def get_stored_prompts_for_comfyui(self) -> List[str]:
        """
        Get all stored image prompts ready for ComfyUI batch processing
        
        Returns:
            List of image prompts
        """
        return self.image_prompts.copy()
    
    def get_image_timestamps(self) -> List[float]:
        """
        Get all image timestamps for video synchronization
        
        Returns:
            List of timestamps in seconds
        """
        return self.image_timestamps.copy()
    
    def process_user_input_realtime(self, user_input: str, video_duration: int = 60) -> Dict:
        """
        Main real-time processing method: takes user input and generates everything needed
        
        Args:
            user_input: User's input text
            video_duration: Target video duration in seconds
            
        Returns:
            Dictionary with all generated data for video creation
        """
        print(f"🚀 Processing user input in real-time mode...")
        print(f"📝 Input: {user_input[:100]}...")
        print(f"⏱️ Target duration: {video_duration}s")
        
        try:
            # Generate images with timestamps
            image_data = self.generate_realtime_images(user_input, video_duration)
            
            # Prepare data for ComfyUI and video generation
            result = {
                "user_input": user_input,
                "art_style": self.selected_art_style,
                "video_duration": video_duration,
                "total_images": len(image_data),
                "image_data": image_data,
                "comfyui_prompts": self.get_stored_prompts_for_comfyui(),
                "timestamps": self.get_image_timestamps()
            }
            
            print(f"✅ Real-time processing complete!")
            print(f"   - Art style: {result['art_style']}")
            print(f"   - Images: {result['total_images']}")
            print(f"   - Prompts ready for ComfyUI: {len(result['comfyui_prompts'])}")
            
            return result
            
        except Exception as e:
            print(f"❌ Real-time processing failed: {e}")
            raise
    
    def batch_generate_images_with_comfyui(self, image_data: List[Dict]) -> List[str]:
        """
        Send all image prompts to ComfyUI for batch generation
        
        Args:
            image_data: List of image data dictionaries from generate_realtime_images
            
        Returns:
            List of generated image file paths
        """
        print(f"🎨 Sending {len(image_data)} prompts to ComfyUI for batch generation...")
        
        generated_paths = []
        output_folder = f"realtime_gen_{int(time.time())}"
        
        for i, data in enumerate(image_data):
            try:
                print(f"🖼️ Generating image {i+1}/{len(image_data)} (t={data['timestamp']:.1f}s)...")
                
                # Submit to ComfyUI
                prompt_id = call_image_request_function(data['prompt'], f"{output_folder}_{i}")
                
                if prompt_id:
                    print(f"✓ Submitted to ComfyUI: {prompt_id}")
                    
                    # Check completion with heartbeat
                    if self.check_comfyui_completion_heartbeat(prompt_id, f"{output_folder}_{i}", max_attempts=15):
                        
                        # Get generated files
                        generated_files = self.get_generated_files(prompt_id, f"{output_folder}_{i}")
                        image_files = [f for f in generated_files if any(ext in f.lower() for ext in ['.png', '.jpg', '.jpeg'])]
                        
                        if image_files:
                            generated_paths.append(image_files[0])
                            print(f"✅ Image {i+1} ready: {image_files[0]}")
                        else:
                            print(f"⚠️ No image found for prompt {i+1}")
                            generated_paths.append(None)
                    else:
                        print(f"⚠️ Timeout for image {i+1}")
                        generated_paths.append(None)
                else:
                    print(f"❌ Failed to submit image {i+1}")
                    generated_paths.append(None)
                    
            except Exception as e:
                print(f"❌ Error generating image {i+1}: {e}")
                generated_paths.append(None)
        
        successful = len([p for p in generated_paths if p is not None])
        print(f"✅ Batch generation complete: {successful}/{len(image_data)} images generated")
        
        return generated_paths
    
    def reset_realtime_state(self):
        """Reset the real-time generation state for new session"""
        self.selected_art_style = None
        self.image_prompts.clear()
        self.image_timestamps.clear()
        print("🔄 Real-time state reset")

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        if hasattr(self, 'video_generator'):
            self.video_generator.cleanup()


def main():
    """Example usage of the RealtimeVideoGenerator with new real-time features"""
    
    # Sample user inputs for real-time video generation
    user_input_1 = """
    Artificial Intelligence is transforming our world at an unprecedented pace.
    From self-driving cars to medical diagnosis, AI is revolutionizing every industry.
    Machine learning algorithms can now understand language, recognize images, and even create art.
    The future of AI holds incredible possibilities for humanity.
    """
    
    user_input_2 = """
    Once upon a time, in a magical forest, there lived a friendly dragon who loved to read books.
    Every day, the dragon would visit the village library and share stories with the children.
    The dragon's favorite tales were about brave knights and beautiful princesses.
    """
    
    # Initialize generator with ComfyUI URL (update this with your actual URL)
    comfyui_url = "https://7fd6781ec07e.ngrok-free.app/api/prompt"  # Update this!
    generator = RealtimeVideoGenerator(comfyui_url=comfyui_url)
    
    try:
        # Example 1: Real-time image generation with timestamps
        print("=== Example 1: Real-Time Image Generation ===\n")
        
        # Process first user input
        result1 = generator.process_user_input_realtime(user_input_1, video_duration=20)
        print(f"\n📊 Real-time Result 1:")
        print(f"   Art Style: {result1['art_style']}")
        print(f"   Total Images: {result1['total_images']}")
        print(f"   Timestamps: {result1['timestamps']}")
        print(f"   ComfyUI Prompts Ready: {len(result1['comfyui_prompts'])}")
        
        # Process second user input (will reuse art style if similar content)
        print(f"\n--- Processing Second Input ---")
        result2 = generator.process_user_input_realtime(user_input_2, video_duration=15)
        print(f"\n📊 Real-time Result 2:")
        print(f"   Art Style: {result2['art_style']}")
        print(f"   Total Images: {result2['total_images']}")
        print(f"   Timestamps: {result2['timestamps']}")
        
        # Example 2: Batch generate images with ComfyUI
        print("\n=== Example 2: Batch ComfyUI Generation ===\n")
        
        # Generate images for first result
        print("🎨 Generating images for first input...")
        generated_images = generator.batch_generate_images_with_comfyui(result1['image_data'])
        print(f"✅ Generated {len([img for img in generated_images if img])} images")
        
        # Example 3: Show stored prompts for ComfyUI
        print("\n=== Example 3: Stored Prompts for ComfyUI ===\n")
        
        stored_prompts = generator.get_stored_prompts_for_comfyui()
        timestamps = generator.get_image_timestamps()
        
        print(f"📝 Stored {len(stored_prompts)} prompts:")
        for i, (prompt, timestamp) in enumerate(zip(stored_prompts[:3], timestamps[:3])):
            print(f"   {i+1}. [{timestamp:.1f}s] {prompt[:80]}...")
        
        # Example 4: Reset and try different content
        print("\n=== Example 4: Reset State and New Content ===\n")
        
        generator.reset_realtime_state()
        
        educational_input = """
        The water cycle is a fascinating natural process that affects all life on Earth.
        Water evaporates from oceans and lakes, forms clouds in the atmosphere,
        and returns to Earth as precipitation, continuing the endless cycle.
        """
        
        result3 = generator.process_user_input_realtime(educational_input, video_duration=25)
        print(f"\n📊 Educational Content Result:")
        print(f"   New Art Style: {result3['art_style']}")
        print(f"   Images: {result3['total_images']}")
        
        # Example 5: Legacy pipeline for comparison
        print("\n=== Example 5: Legacy Pipeline (for comparison) ===\n")
        
        video_path = generator.process_article_to_video(
            article=user_input_1,
            output_path="realtime_comparison_video.mp4",
            target_duration=20,
            use_comfyui=False,  # Use existing images for speed
            aspect_ratio="9:16"
        )
        print(f"✅ Legacy video created: {video_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        generator.cleanup()


if __name__ == "__main__":
    main()