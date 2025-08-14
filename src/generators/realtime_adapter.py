#!/usr/bin/env python3
"""
Adapter for RealtimeVideoGenerator to match VideoGenerator interface
"""

import os
import tempfile
import librosa
from typing import List, Tuple, Dict
from .Realtime_Video_Gen import RealtimeVideoGenerator as RealTimeGen
from .video_generator import VideoGenerator

class RealtimeVideoGeneratorAdapter:
    """Adapter to make RealtimeVideoGenerator compatible with VideoGenerator interface"""
    
    def __init__(self, comfyui_url: str = None):
        """Initialize the adapter with RealtimeVideoGenerator
        
        Args:
            comfyui_url: Optional ComfyUI API endpoint URL
        """
        # Initialize with ComfyUI URL if provided
        self.realtime_gen = RealTimeGen(comfyui_url=comfyui_url)
        # The realtime generator already has a video_gen instance
        self.video_gen = self.realtime_gen.video_generator
        self.temp_dir = tempfile.mkdtemp()
    
    def extract_web_content(self, url: str) -> str:
        """Extract content from web URL - delegate to VideoGenerator"""
        return self.video_gen.extract_web_content(url)
    
    def generate_transcript_from_article(self, article_text: str) -> str:
        """Generate transcript from article - delegate to VideoGenerator"""
        return self.video_gen.generate_transcript_from_article(article_text)
    
    def generate_audio_from_api(self, text: str) -> Tuple[str, float]:
        """Generate audio from text - delegate to VideoGenerator"""
        return self.video_gen.generate_audio_from_api(text)
    
    def get_image_list_script(self, transcript: str, duration: float) -> List[Dict]:
        """
        Generate image descriptions using RealtimeVideoGenerator's scene breakdown
        Returns a list of image descriptions with durations
        """
        try:
            # Use RealtimeVideoGenerator's new scene breakdown method
            return self.realtime_gen.break_script_into_scenes(transcript, int(duration))
            
        except Exception as e:
            print(f"Error generating image script with realtime generator: {e}")
            # Fallback to VideoGenerator method
            return self.video_gen.get_image_list_script(transcript, duration)
    
    def vector_search_images(self, descriptions: List[str]) -> List[str]:
        """Search for images based on descriptions - delegate to VideoGenerator"""
        return self.video_gen.vector_search_images(descriptions)
    
    def _collect_image_previews(self, image_paths: List[str], descriptions: List[str]) -> List[Dict]:
        """Collect image previews - delegate to VideoGenerator"""
        return self.video_gen._collect_image_previews(image_paths, descriptions)
    
    def create_video_from_prepared_data_with_images(
        self, 
        transcript: str, 
        audio_path: str, 
        image_paths: List[str], 
        descriptions: List[str],
        output_path: str = None,
        use_blur_background: bool = False,
        overlay_text: str = "",
        blur_strength: int = 24,
        aspect_ratio: str = "9:16"
    ) -> str:
        """Create video from prepared data - delegate to VideoGenerator"""
        return self.video_gen.create_video_from_prepared_data_with_images(
            transcript, audio_path, image_paths, descriptions,
            output_path, use_blur_background, overlay_text, blur_strength, aspect_ratio
        )
    
    def create_video_from_prepared_data(
        self,
        transcript: str,
        audio_path: str,
        image_script: List[Dict],
        output_path: str = None,
        use_blur_background: bool = False,
        overlay_text: str = "",
        blur_strength: int = 24,
        aspect_ratio: str = "9:16"
    ) -> str:
        """Create video from prepared data - delegate to VideoGenerator"""
        return self.video_gen.create_video_from_prepared_data(
            transcript, audio_path, image_script,
            output_path, use_blur_background, overlay_text, blur_strength, aspect_ratio
        )
    
    def generate_video_from_article(
        self,
        article_text: str,
        topic: str = None,
        output_path: str = None,
        use_gpt_transcript: bool = True,
        use_comfyui: bool = False,
        target_duration: int = 60,
        aspect_ratio: str = "9:16"
    ) -> str:
        """Generate video from article using RealtimeVideoGenerator's integrated pipeline"""
        try:
            if output_path is None:
                import time
                output_path = f"realtime_video_{int(time.time())}.mp4"
            
            # Use the integrated pipeline method
            return self.realtime_gen.process_article_to_video(
                article=article_text,
                output_path=output_path,
                target_duration=target_duration,
                use_comfyui=use_comfyui,  # Allow choice of ComfyUI vs database images
                aspect_ratio=aspect_ratio
            )
            
        except Exception as e:
            print(f"Error with realtime pipeline, falling back to standard: {e}")
            # Fallback to standard VideoGenerator
            return self.video_gen.generate_video_from_article(
                article_text, topic, output_path, use_gpt_transcript
            )
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'realtime_gen'):
            self.realtime_gen.cleanup()
        if hasattr(self, 'video_gen') and hasattr(self.video_gen, 'cleanup'):
            # VideoGenerator might not have cleanup method
            try:
                self.video_gen.cleanup()
            except:
                pass