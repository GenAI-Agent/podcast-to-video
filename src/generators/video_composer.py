#!/usr/bin/env python3
"""
Video Composer Module
Handles final video composition from script, audio, images, and timeline
"""

import os
import sys
import tempfile
import subprocess
import re
from typing import List, Dict, Optional
import librosa

# Add project root to sys.path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

class VideoComposer:
    def __init__(self):
        """Initialize the video composer"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Default video settings
        self.default_width = 1080
        self.default_height = 1920
        self.default_fps = 30
        
        # Subtitle settings
        self.subtitle_font_size = 1.0  # Percentage of height
        self.subtitle_position = "center"
        self.subtitle_margin = 30  # Percentage from bottom
    
    def compose_video(
        self,
        script: str,
        audio_path: str,
        timeline: List[Dict],
        output_path: str,
        settings: Dict = None
    ) -> str:
        """
        Compose final video from all components
        
        Args:
            script: Video script text
            audio_path: Path to audio file
            timeline: Image timeline with assignments
            output_path: Output video file path
            settings: Video composition settings
            
        Returns:
            Path to generated video file
        """
        settings = settings or {}
        
        try:
            print("🎬 Starting video composition...")
            print(f"📝 Script length: {len(script)} characters")
            print(f"🎵 Audio file: {audio_path}")
            print(f"🖼️ Timeline segments: {len(timeline)}")
            
            # Get audio duration
            audio_duration = librosa.get_duration(path=audio_path)
            print(f"🎵 Audio duration: {audio_duration:.2f} seconds")
            
            # Apply settings
            aspect_ratio = settings.get("aspect_ratio", "9:16")
            use_blur_background = settings.get("use_blur_background", False)
            overlay_text = settings.get("overlay_text", "")
            blur_strength = settings.get("blur_strength", 24)
            
            # Set video dimensions based on aspect ratio
            if aspect_ratio == "16:9":
                width, height = 1920, 1080
            else:  # Default to 9:16 (vertical)
                width, height = self.default_width, self.default_height
            
            print(f"📐 Video dimensions: {width}x{height}")
            
            # Generate SRT subtitles from script
            srt_path = self._generate_srt_file(script, audio_duration)
            print(f"📄 Generated SRT: {srt_path}")
            
            # Prepare images for video
            image_list = self._prepare_image_list(timeline, audio_duration)
            
            # Create video based on settings
            if use_blur_background:
                print("🌫️ Creating video with blur background effect...")
                video_path = self._create_video_with_blur_background(
                    image_list, timeline, audio_path, srt_path, output_path,
                    audio_duration, overlay_text, blur_strength, aspect_ratio
                )
            else:
                print("🎬 Creating standard video...")
                video_path = self._create_standard_video(
                    image_list, audio_path, srt_path, output_path, audio_duration
                )
            
            print(f"✅ Video composition complete: {video_path}")
            return video_path
            
        except Exception as e:
            raise Exception(f"Failed to compose video: {str(e)}")
    
    def _generate_srt_file(self, script: str, audio_duration: float) -> str:
        """Generate SRT subtitle file from script"""
        # Split script into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', script) if s.strip()]
        
        if not sentences:
            sentences = [script]
        
        # Calculate timing for each sentence
        time_per_sentence = audio_duration / len(sentences)
        
        srt_content = ""
        for i, sentence in enumerate(sentences):
            start_time = i * time_per_sentence
            end_time = (i + 1) * time_per_sentence
            
            start_formatted = self._format_srt_time(start_time)
            end_formatted = self._format_srt_time(end_time)
            
            srt_content += f"{i + 1}\n"
            srt_content += f"{start_formatted} --> {end_formatted}\n"
            srt_content += f"{sentence}\n\n"
        
        # Save to temporary file
        srt_path = os.path.join(self.temp_dir, "subtitles.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        return srt_path
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time in SRT format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _prepare_image_list(self, timeline: List[Dict], audio_duration: float) -> List[str]:
        """Prepare list of image paths for video creation"""
        image_paths = []
        
        for item in timeline:
            if item.get("image") and item["image"].get("file_path"):
                image_path = item["image"]["file_path"]
                # Convert to WSL path if needed
                wsl_path = self._convert_windows_path_to_wsl(image_path)
                
                # Verify file exists
                if os.path.exists(wsl_path):
                    image_paths.append(wsl_path)
                else:
                    print(f"⚠ Image not found: {image_path}")
                    # Use fallback or skip
                    image_paths.append(None)
            else:
                print(f"⚠ No image assigned for timeline item {item.get('index', '?')}")
                image_paths.append(None)
        
        return image_paths
    
    def _convert_windows_path_to_wsl(self, windows_path: str) -> str:
        """Convert Windows path to WSL path"""
        if windows_path.startswith("C:"):
            return windows_path.replace("C:", "/mnt/c").replace("\\", "/")
        return windows_path
    
    def _create_standard_video(
        self, 
        image_paths: List[str], 
        audio_path: str, 
        srt_path: str, 
        output_path: str, 
        audio_duration: float
    ) -> str:
        """Create standard video without special effects"""
        
        try:
            # Filter out None values and create image list
            valid_images = [img for img in image_paths if img and os.path.exists(img)]
            
            if not valid_images:
                raise Exception("No valid images available for video creation")
            
            # Create image list file for FFmpeg
            image_list_path = os.path.join(self.temp_dir, "image_list.txt")
            
            # Calculate duration per image
            duration_per_image = audio_duration / len(valid_images)
            
            with open(image_list_path, "w") as f:
                for image_path in valid_images:
                    f.write(f"file '{image_path}'\n")
                    f.write(f"duration {duration_per_image:.2f}\n")
                
                # Repeat last image for final frame
                if valid_images:
                    f.write(f"file '{valid_images[-1]}'\n")
            
            # FFmpeg command for standard video
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", image_list_path,
                "-i", audio_path,
                "-vf", f"scale={self.default_width}:{self.default_height}:force_original_aspect_ratio=decrease,pad={self.default_width}:{self.default_height}:(ow-iw)/2:(oh-ih)/2,subtitles={srt_path}:force_style='FontSize={int(self.default_height * self.subtitle_font_size / 100)},Alignment=2,MarginV={int(self.default_height * self.subtitle_margin / 100)}'",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-shortest",
                "-r", str(self.default_fps),
                output_path
            ]
            
            print("🎬 Executing FFmpeg for standard video...")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg failed: {result.stderr}")
            
            print(f"✅ Standard video created: {output_path}")
            return output_path
            
        except Exception as e:
            raise Exception(f"Failed to create standard video: {str(e)}")
    
    def _create_video_with_blur_background(
        self,
        image_paths: List[str],
        timeline: List[Dict],
        audio_path: str,
        srt_path: str,
        output_path: str,
        audio_duration: float,
        overlay_text: str = "",
        blur_strength: int = 24,
        aspect_ratio: str = "9:16"
    ) -> str:
        """Create video with blur background effect"""
        
        try:
            print("🌫️ Creating video with blur background effect...")
            
            # Set dimensions based on aspect ratio
            if aspect_ratio == "16:9":
                width, height = 1920, 1080
            else:
                width, height = 1080, 1920
            
            print(f"📐 Aspect ratio: {aspect_ratio} ({width}x{height})")
            print(f"🌫️ Blur strength: {blur_strength}")
            
            # Filter valid images and match with timeline
            valid_segments = []
            for i, (image_path, timeline_item) in enumerate(zip(image_paths, timeline)):
                if image_path and os.path.exists(image_path):
                    valid_segments.append({
                        "image_path": image_path,
                        "duration": timeline_item.get("duration", 3.0)
                    })
                else:
                    print(f"⚠ Skipping invalid image for segment {i}")
            
            if not valid_segments:
                raise Exception("No valid images available for blur video creation")
            
            # Create image effects list for FFmpeg
            effects_list_path = os.path.join(self.temp_dir, "images_blur_effect.txt")
            
            total_duration = 0
            with open(effects_list_path, "w") as f:
                for segment in valid_segments:
                    duration = segment["duration"]
                    f.write(f"file '{segment['image_path']}'\n")
                    f.write(f"duration {duration:.2f}\n")
                    total_duration += duration
                
                # Repeat last image
                if valid_segments:
                    f.write(f"file '{valid_segments[-1]['image_path']}'\n")
            
            print(f"📊 Video timing: {len(valid_segments)} images with blur background effect")
            print(f"📊 Image total duration: {total_duration:.2f}s")
            print(f"📊 Audio duration: {audio_duration:.2f}s")
            
            # Create temporary video with blur effect
            temp_blur_video = os.path.join(self.temp_dir, "temp_video_blur.mp4")
            
            # Complex filter for blur background effect
            blur_filter = (
                f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2[scaled];"
                f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height}[cropped];"
                f"[cropped]gblur=sigma={blur_strength}[blurred];"
                f"[blurred][scaled]overlay=(W-w)/2:(H-h)/2"
            )
            
            if overlay_text:
                # Add text overlay
                text_filter = f",drawtext=text='{overlay_text}':fontcolor=white:fontsize={int(height * 0.04)}:x=(w-text_w)/2:y=h*0.1"
                blur_filter += text_filter
            
            # Create blur video
            blur_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", effects_list_path,
                "-i", audio_path,
                "-filter_complex", blur_filter,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-crf", "18",
                "-r", "30",
                "-shortest",
                temp_blur_video
            ]
            
            print("🎬 Executing FFmpeg with blur background effect...")
            result = subprocess.run(blur_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Blur video creation failed: {result.stderr}")
            
            # Add subtitles to final video
            final_cmd = [
                "ffmpeg", "-y",
                "-i", temp_blur_video,
                "-vf", f"subtitles={srt_path}:force_style='FontSize={int(height * self.subtitle_font_size / 100)},Alignment=2,MarginV={int(height * self.subtitle_margin / 100)},Bold=1,BorderStyle=1,Outline=2,OutlineColour=&H000000&,PrimaryColour=&HFFFFFF&'",
                "-c:a", "copy",
                "-crf", "23",
                output_path
            ]
            
            result = subprocess.run(final_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Subtitle overlay failed: {result.stderr}")
            
            print(f"✅ Successfully created blur background video: {output_path}")
            print(f"📐 Format: {aspect_ratio} ({width}x{height})")
            print(f"🌫️ Blur effect applied with strength {blur_strength}")
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Failed to create blur background video: {str(e)}")
    
    def get_composition_settings(self) -> Dict:
        """Get available composition settings and their options"""
        return {
            "aspect_ratios": [
                {"value": "9:16", "label": "9:16 (Vertical/Mobile)", "width": 1080, "height": 1920},
                {"value": "16:9", "label": "16:9 (Horizontal/Desktop)", "width": 1920, "height": 1080}
            ],
            "blur_effects": [
                {"value": False, "label": "Standard Video"},
                {"value": True, "label": "Blur Background Effect"}
            ],
            "blur_strengths": [
                {"value": 12, "label": "Light Blur"},
                {"value": 24, "label": "Medium Blur"},
                {"value": 36, "label": "Heavy Blur"}
            ],
            "subtitle_positions": [
                {"value": "bottom", "label": "Bottom"},
                {"value": "center", "label": "Center"},
                {"value": "top", "label": "Top"}
            ]
        }
    
    def validate_composition_inputs(
        self, 
        script: str, 
        audio_path: str, 
        timeline: List[Dict]
    ) -> Dict:
        """Validate inputs for video composition"""
        result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Validate script
        if not script or not script.strip():
            result["errors"].append("Script is empty")
            result["valid"] = False
        
        # Validate audio
        if not audio_path or not os.path.exists(audio_path):
            result["errors"].append("Audio file not found")
            result["valid"] = False
        
        # Validate timeline
        if not timeline or len(timeline) == 0:
            result["errors"].append("Timeline is empty")
            result["valid"] = False
        else:
            # Check for missing images
            missing_images = len([item for item in timeline if not item.get("image")])
            if missing_images > 0:
                result["warnings"].append(f"{missing_images} timeline segments have no assigned images")
            
            # Check timeline duration consistency
            total_timeline_duration = sum(item.get("duration", 0) for item in timeline)
            if audio_path and os.path.exists(audio_path):
                try:
                    audio_duration = librosa.get_duration(path=audio_path)
                    duration_diff = abs(total_timeline_duration - audio_duration)
                    if duration_diff > 2:  # Allow 2 second tolerance
                        result["warnings"].append(f"Timeline duration ({total_timeline_duration:.1f}s) doesn't match audio duration ({audio_duration:.1f}s)")
                except:
                    result["warnings"].append("Could not verify audio duration")
        
        return result