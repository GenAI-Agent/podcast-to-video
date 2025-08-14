#!/usr/bin/env python3
"""
FFmpeg Video Generator Script
Generates video from article text with SRT subtitles, audio, and vector-searched images
"""

import os
import sys
import subprocess
import tempfile
from typing import List, Tuple
import librosa
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# Add project root to sys.path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.generators.text_content_manager import TextContentManager
from src.generators.srt_manager import SRTManager
from src.generators.image_manager import ImageManager

# Load environment variables
load_dotenv()

class VideoGenerator:
    def __init__(self):
        """Initialize the video generator"""
        self.temp_dir = tempfile.mkdtemp()
        # self.temp_dir = "temp"
        self.text_manager = TextContentManager()
        self.srt_manager = SRTManager(self.temp_dir)
        self.image_manager = ImageManager()

    def get_transcript(self, article: str, web_link: str, use_gpt_transcript: bool = True) -> str:
        """Get transcript from article"""
                # Need article or web_link for default flow
        if web_link:
            print("Step 0a: Extracting content from web link...")
            article = self.text_manager.extract_web_content(web_link)
            print(f"Extracted article length: {len(article)} characters")
        elif not article:
            raise ValueError(
                "Either article content or web_link is required when not using custom SRT"
            )

        # Optional: Generate transcript from article using GPT
        if use_gpt_transcript:
            print("Step 0b: Generating transcript from article using GPT...")
            transcript = self.text_manager.generate_transcript_from_article(article)
            print("Transcript generated successfully")
        else:
            transcript = article
        return transcript
    
    def generate_audio_from_api(self, text: str) -> Tuple[str, float]:
        """Generate audio from text using ElevenLabs API"""
        try:
            # Get ElevenLabs API credentials from environment
            elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
            if not elevenlabs_api_key:
                raise ValueError("ELEVENLABS_API_KEY environment variable not set")

            # Initialize ElevenLabs client
            elevenlabs = ElevenLabs(api_key=elevenlabs_api_key)

            # Voice configuration - using JBFqnCBsd6RMkjVDRZzb as in your example
            # voice_id = "JBFqnCBsd6RMkjVDRZzb" # for English
            voice_id = "fQj4gJSexpu8RDE2Ii5m" # for Chinese

            print(
                f"Generating audio with ElevenLabs API (text length: {len(text)} chars)..."
            )

            # Generate audio using the client
            audio = elevenlabs.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )

            # Save audio to temp directory
            temp_audio_path = os.path.join(self.temp_dir, "generated_audio.mp3")
            with open(temp_audio_path, "wb") as f:
                # audio is an iterator, so we need to iterate through the chunks
                for chunk in audio:
                    f.write(chunk)

            # Get audio duration
            duration = librosa.get_duration(path=temp_audio_path)
            print(f"✓ Generated audio from ElevenLabs API, duration: {duration:.2f}s")

            return temp_audio_path, duration

        except Exception as e:
            print(f"Error generating audio from ElevenLabs API: {e}")
            raise

    def generate_audio(
        self, custom_audio_path: str = None, text_for_generation: str = None
    ) -> Tuple[str, float]:
        """Process audio file and return path and duration"""

        # If custom audio path is provided, use it
        if custom_audio_path and os.path.exists(custom_audio_path):
            print(f"Using custom audio file: {custom_audio_path}")

            # Get audio duration using multiple methods for reliability
            duration = librosa.get_duration(path=custom_audio_path)
            print(f"Librosa detected duration: {duration:.2f}s")

            # Also check with ffprobe for verification
            try:
                ffprobe_cmd = [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    custom_audio_path,
                ]
                result = subprocess.run(
                    ffprobe_cmd, capture_output=True, text=True, check=True
                )
                ffprobe_duration = float(result.stdout.strip())
                print(f"FFprobe detected duration: {ffprobe_duration:.2f}s")

                # Use ffprobe duration if significantly different
                if abs(duration - ffprobe_duration) > 1.0:
                    print(
                        f"Duration mismatch detected, using FFprobe duration: {ffprobe_duration:.2f}s"
                    )
                    duration = ffprobe_duration
            except Exception as e:
                print(f"FFprobe duration check failed: {e}, using librosa duration")

            # Convert to MP3 format to ensure compatibility
            temp_audio_path = os.path.join(self.temp_dir, "audio.mp3")

            # Convert audio to MP3 format with specific settings
            convert_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                custom_audio_path,
                "-acodec",
                "mp3",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-b:a",
                "192k",
                temp_audio_path,
            ]

            subprocess.run(convert_cmd, check=True)
            print(f"✓ Converted audio to MP3 format: {temp_audio_path}")

            # Verify the converted file duration
            converted_duration = librosa.get_duration(path=temp_audio_path)
            print(f"Converted file duration: {converted_duration:.2f}s")

            return temp_audio_path, duration

        # If text is provided, generate audio from API
        elif text_for_generation:
            print("Generating audio from text using API...")
            return self.generate_audio_from_api(text_for_generation)

    def create_video_with_image_script(
        self,
        image_paths: List[str],
        image_script: List[dict],
        audio_path: str,
        srt_path: str,
        output_path: str,
        audio_duration: float,
        aspect_ratio: str = "vertical",  # "vertical" for vertical, "horizontal" for horizontal
    ) -> str:
        """Create video using image script with specific durations for each image"""
        if aspect_ratio == "vertical":
            width, height = 1080, 1920  # Vertical format (mobile/social media)
        else:
            width, height = 1920, 1080  # Horizontal format (traditional)
        
        # Filter and process image paths
        valid_images = []
        used_fallbacks = set()
        
        for img in image_paths:
            if img:
                # Convert Windows path to WSL path and ensure absolute path
                wsl_path = self.image_manager.convert_windows_path_to_wsl(img)
                if os.path.exists(wsl_path):
                    abs_path = os.path.abspath(wsl_path)
                    valid_images.append(abs_path)
                    print(f"✓ Found image: {abs_path}")
                else:
                    print(f"⚠ Original image not found: {wsl_path}")
                    fallback = self.image_manager.find_fallback_image(img, used_fallbacks)
                    if fallback:
                        abs_fallback = os.path.abspath(fallback)
                        valid_images.append(abs_fallback)
                        print(f"✓ Using fallback instead: {abs_fallback}")
                    else:
                        print(f"✗ No fallback found for: {img}")
                        valid_images.append(None)
            else:
                valid_images.append(None)
        
        # Filter out None values
        valid_script_pairs = [(img, script) for img, script in zip(valid_images, image_script) if img is not None]
        
        if not valid_script_pairs:
            raise ValueError("No valid images found for video generation")

        # Create image list file with specific durations
        image_list_path = os.path.join(self.temp_dir, "images_with_durations.txt")
        total_image_duration = 0
        with open(image_list_path, "w") as f:
            for i, (img_path, script_item) in enumerate(valid_script_pairs):
                duration = script_item["duration"]
                total_image_duration += duration
                f.write(f"file '{img_path}'\n")
                f.write(f"duration {duration}\n")
            # Add the last image without duration to ensure proper ending
            if valid_script_pairs:
                f.write(f"file '{valid_script_pairs[-1][0]}'\n")

        print(
            f"📊 Video timing: {len(valid_script_pairs)} images with custom durations"
        )
        print(f"📊 Image total duration: {total_image_duration:.2f}s")
        print(f"📊 Audio duration: {audio_duration:.2f}s")
        if abs(total_image_duration - audio_duration) > 0.1:
            print(f"⚠️  Duration mismatch: {abs(total_image_duration - audio_duration):.2f}s difference")

        # Create main video from images with custom durations
        temp_video_path = os.path.join(self.temp_dir, "temp_video.mp4")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            image_list_path,
            "-i",
            audio_path,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            "-t",
            str(audio_duration),  # Limit video to audio duration
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            temp_video_path,
        ]

        subprocess.run(ffmpeg_cmd, check=True)

        # Add subtitles
        video_with_subs = os.path.join(self.temp_dir, "video_with_subs.mp4")
        
        # Different alignment for vertical vs horizontal
        if aspect_ratio == "vertical":
            # Vertical: top center
            alignment = 8  # Top center
            margin_v = 50  # Top margin
        else:
            # Horizontal: bottom center  
            alignment = 2  # Bottom center
            margin_v = 20  # Bottom margin - reduced for horizontal

        subtitle_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_video_path,
            "-vf",
            f"subtitles={srt_path}:force_style='FontSize=12,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=3,Alignment={alignment},MarginV={margin_v},Bold=1'",
            "-c:a",
            "copy",
            video_with_subs,
        ]

        subprocess.run(subtitle_cmd, check=True)

        # Add ending video (same logic as original)
        local_ending_video = os.path.join(os.getcwd(), "data/media/LensCover.mp4")
        wsl_ending_video = "/mnt/c/Users/x7048/Documents/VideoMaker/LensCover.mp4"

        ending_video = None
        if os.path.exists(local_ending_video):
            ending_video = local_ending_video
            print(f"✓ Using local ending video: {local_ending_video}")
        elif os.path.exists(wsl_ending_video):
            ending_video = wsl_ending_video
            print(f"✓ Using WSL ending video: {wsl_ending_video}")
        else:
            print("⚠ No ending video found")

        if ending_video:
            # Re-encode ending video to match main video format
            temp_ending = os.path.join(self.temp_dir, "ending_25fps.mp4")

            convert_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                ending_video,
                "-r",
                "25",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-ar",
                "44100",
                "-ac",
                "2",
                temp_ending,
            ]

            subprocess.run(convert_cmd, check=True)

            # Concat with matching formats
            final_video_list = os.path.join(self.temp_dir, "final_list.txt")
            with open(final_video_list, "w") as f:
                f.write(f"file '{video_with_subs}'\n")
                f.write(f"file '{temp_ending}'\n")

            final_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                final_video_list,
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-avoid_negative_ts",
                "make_zero",
                "-fflags",
                "+genpts",
                output_path,
            ]

            subprocess.run(final_cmd, check=True)
        else:
            # Just copy the video with subtitles as final output
            subprocess.run(["cp", video_with_subs, output_path], check=True)

        return output_path

    def create_video_with_blur_background(
        self,
        image_paths: List[str],
        image_script: List[dict],
        audio_path: str,
        srt_path: str,
        output_path: str,
        audio_duration: float,
        blur_strength: int = 24,
        aspect_ratio: str = "vertical",  # "vertical" for vertical, "horizontal" for horizontal
    ) -> str:
        """Create video with blur background effect
        
        This method creates a cinematic blur background by:
        1. Scaling the image to fill the entire frame (eliminating black bars)
        2. Applying heavy Gaussian blur to the scaled background
        3. Overlaying the sharp original image centered on top
        
        Args:
            image_paths: List of image file paths
            image_script: List of dictionaries with image prompts and durations
            audio_path: Path to audio file
            srt_path: Path to SRT subtitle file
            output_path: Path for output video
            audio_duration: Duration of audio in seconds
            blur_strength: Gaussian blur strength (default: 24)
            aspect_ratio: Video aspect ratio - "9:16" for vertical, "16:9" for horizontal
        """
        # Set dimensions based on aspect ratio
        if aspect_ratio == "vertical":
            width, height = 1080, 1920  # Vertical format (mobile/social media)
        else:
            width, height = 1920, 1080  # Horizontal format (traditional)
        
        print(f"🎬 Creating video with repeating blur background effect")
        print(f"📐 Aspect ratio: {aspect_ratio} ({width}x{height})")
        print(f"🌫️  Blur strength: {blur_strength}")
        
        # Ensure we have the same number of images and script items
        if len(image_paths) != len(image_script):
            print(f"Warning: Mismatch between images ({len(image_paths)}) and script items ({len(image_script)})")
            # Adjust script to match images
            if len(image_paths) > len(image_script):
                # Add default durations for extra images
                avg_duration = audio_duration / len(image_paths)
                for i in range(len(image_script), len(image_paths)):
                    image_script.append({"description": f"image_{i}", "duration": avg_duration})
            else:
                # Truncate script to match available images
                image_script = image_script[:len(image_paths)]

        # Filter and process image paths
        valid_images = []
        used_fallbacks = set()
        
        for img in image_paths:
            if img:
                # Convert Windows path to WSL path and ensure absolute path
                wsl_path = self.image_manager.convert_windows_path_to_wsl(img)
                if os.path.exists(wsl_path):
                    abs_path = os.path.abspath(wsl_path)
                    valid_images.append(abs_path)
                    print(f"✓ Found image: {abs_path}")
                else:
                    print(f"⚠ Original image not found: {wsl_path}")
                    fallback = self.image_manager.find_fallback_image(img, used_fallbacks)
                    if fallback:
                        abs_fallback = os.path.abspath(fallback)
                        valid_images.append(abs_fallback)
                        print(f"✓ Using fallback instead: {abs_fallback}")
                    else:
                        print(f"✗ No fallback found for: {img}")
                        valid_images.append(None)
            else:
                valid_images.append(None)
        
        # Filter out None values
        valid_script_pairs = [(img, script) for img, script in zip(valid_images, image_script) if img is not None]
        
        if not valid_script_pairs:
            raise ValueError("No valid images found for video generation")

        # Create image list file with specific durations
        image_list_path = os.path.join(self.temp_dir, "images_blur_effect.txt")
        total_image_duration = 0
        with open(image_list_path, "w") as f:
            for i, (img_path, script_item) in enumerate(valid_script_pairs):
                duration = script_item["duration"]
                total_image_duration += duration
                f.write(f"file '{img_path}'\n")
                f.write(f"duration {duration}\n")
            # Add the last image without duration to ensure proper ending
            if valid_script_pairs:
                f.write(f"file '{valid_script_pairs[-1][0]}'\n")

        print(f"📊 Video timing: {len(valid_script_pairs)} images with repeating blur background")
        print(f"📊 Image total duration: {total_image_duration:.2f}s")
        print(f"📊 Audio duration: {audio_duration:.2f}s")

        # Create main video with repeating blur background effect
        temp_video_path = os.path.join(self.temp_dir, "temp_video_blur.mp4")
        
        # Enhanced filter that creates blur background:
        # 1. Scale the image to fill the entire frame (no black bars)
        # 2. Apply heavy blur to the scaled background for cinematic effect
        # 3. Scale original image to fit within frame while maintaining aspect ratio  
        # 4. Center the sharp original image on the blurred background
        
        filter_complex = (
            # Create blurred background by scaling image to fill entire frame
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase[bg_scaled];"
            f"[bg_scaled]crop={width}:{height}:(iw-ow)/2:(ih-oh)/2,gblur={blur_strength}[blurred];"
            # Scale original image to fit within frame while maintaining aspect ratio
            f"[0:v]scale={min(width, height)}:{min(width, height)}:force_original_aspect_ratio=decrease[scaled];"
            # Composite sharp image over blurred background
            f"[blurred][scaled]overlay=(W-w)/2:(H-h)/2[final]"
        )

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", image_list_path,
            "-i", audio_path,
            "-filter_complex", filter_complex,
            "-map", "[final]",
            "-map", "1:a",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-r", "30",
            "-crf", "18",
            "-preset", "medium",
            "-t", str(audio_duration),
            temp_video_path,
        ]

        print("🎬 Executing FFmpeg with repeating blur background effect...")
        subprocess.run(ffmpeg_cmd, check=True)

        # Handle ending video first (same logic as original methods)
        local_ending_video = os.path.join(os.getcwd(), "data/media/LensCover.mp4")
        wsl_ending_video = "/mnt/c/Users/x7048/Documents/VideoMaker/LensCover.mp4"

        ending_video = None
        if os.path.exists(local_ending_video):
            ending_video = local_ending_video
            print(f"✓ Using local ending video: {local_ending_video}")
        elif os.path.exists(wsl_ending_video):
            ending_video = wsl_ending_video
            print(f"✓ Using WSL ending video: {wsl_ending_video}")
        else:
            print("⚠ No ending video found")

        # First create the concatenated video without subtitles
        temp_concat_video = os.path.join(self.temp_dir, "video_blur_concat.mp4")
        
        if ending_video:
            # Re-encode ending video to match main video format and aspect ratio
            temp_ending = os.path.join(self.temp_dir, "ending_blur_format.mp4")
            
            # Scale ending video to match the blur video format
            ending_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"

            convert_cmd = [
                "ffmpeg",
                "-y",
                "-i", ending_video,
                "-vf", ending_filter,
                "-r", "30",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-ar", "44100",
                "-ac", "2",
                temp_ending,
            ]

            subprocess.run(convert_cmd, check=True)

            # Concat with matching formats
            final_video_list = os.path.join(self.temp_dir, "final_blur_list.txt")
            with open(final_video_list, "w") as f:
                f.write(f"file '{temp_video_path}'\n")
                f.write(f"file '{temp_ending}'\n")

            final_cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", final_video_list,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                "-fflags", "+genpts",
                temp_concat_video,
            ]

            subprocess.run(final_cmd, check=True)
        else:
            # Just copy the main video
            subprocess.run(["cp", temp_video_path, temp_concat_video], check=True)

        # NOW add subtitles to the final concatenated video
        # Enhanced subtitle styling positioned based on aspect ratio
        font_size = 12
        
        # Different alignment for vertical vs horizontal
        if aspect_ratio == "vertical":
            # Vertical: top center
            alignment = 8  # Top center
            margin_v = 50  # Top margin
        else:
            # Horizontal: bottom center
            alignment = 2  # Bottom center
            margin_v = 20  # Bottom margin - reduced for horizontal
        
        # Subtitle styling optimized for blur background visibility
        subtitle_style = (
            f"FontSize={font_size},"
            f"PrimaryColour=&Hffffff,"  # White text
            f"OutlineColour=&H000000,"  # Black outline
            f"Outline=3,"               # Thicker outline for better visibility over blur
            f"Alignment={alignment},"   # Position based on aspect ratio
            f"MarginV={margin_v},"      # Margin based on aspect ratio
            f"Bold=1"                   # Bold text
        )

        subtitle_cmd = [
            "ffmpeg",
            "-y",
            "-i", temp_concat_video,
            "-vf", f"subtitles={srt_path}:force_style='{subtitle_style}'",
            "-c:a", "copy",
            output_path,
        ]

        subprocess.run(subtitle_cmd, check=True)

        print(f"✅ Successfully created repeating blur background video: {output_path}")
        print(f"📐 Format: {aspect_ratio} ({width}x{height})")
        print(f"🌫️  Blur background effect applied with strength {blur_strength}")
        print(f"🔄 Background scaled to fill entire frame ({width}x{height}) with no black bars")
        
        return output_path

    def run_video_generator(
        self,
        article: str = None,
        web_link: str = None,
        output_path: str = "output_video.mp4",
        use_gpt_transcript: bool = False,
        custom_srt_path: str = None,
        custom_audio_path: str = None,
        use_blur_background: bool = False,
        blur_strength: int = 24,
        aspect_ratio: str = "vertical",
    ) -> str:
        """Main function to generate video from article

        Args:
            article: The article content (optional if custom_srt_path is provided)
            output_path: Path for the output video
            use_gpt_transcript: Whether to generate a transcript using GPT first
            custom_srt_path: Path to custom SRT file (optional)
            custom_audio_path: Path to custom audio file (optional)
            use_blur_background: Whether to use blur background effect with repeating pattern
            blur_strength: Gaussian blur intensity (default: 24)
            aspect_ratio: Video aspect ratio - "9:16" for vertical, "16:9" for horizontal
        """
        try:
            # Handle SRT generation or use custom SRT
            if custom_srt_path and os.path.exists(custom_srt_path):
                print("Using custom SRT file provided by user...")
                srt_path = custom_srt_path
                # Read sentences from SRT for image search
                sentences = self.srt_manager.extract_sentences_from_srt(custom_srt_path)
            else:
                transcript = self.get_transcript(article, web_link, use_gpt_transcript)
                print("Step 1: Splitting article into sentences...")
                sentences = self.text_manager.split_article_into_sentences(transcript)

            # Handle audio generation or use custom audio
            if custom_audio_path and os.path.exists(custom_audio_path):
                print("Using custom audio file provided by user...")
                audio_path, audio_duration = self.generate_audio(
                    custom_audio_path=custom_audio_path
                )
                print(f"Custom audio duration: {audio_duration:.2f} seconds")
            else:
                print("Step 2: Generating audio from text...")
                # Use the processed transcript for audio generation
                audio_path, audio_duration = self.generate_audio(
                    text_for_generation=transcript
                )
                print(f"Audio duration: {audio_duration:.2f} seconds")

            # Generate SRT if not using custom
            if not custom_srt_path:
                print("Step 3: Generating SRT file...")
                srt_path = self.srt_manager.generate_srt_file(sentences, audio_duration, aspect_ratio)
                print(f"SRT file created: {srt_path}")

            # Generate image list script with durations
            print("Step 4: Generating image list script with prompts...")
            art_style = self.image_manager.select_art_style(transcript)
            image_script = self.image_manager.get_image_list_script(transcript, audio_duration, art_style) # TODO: image generator script
            
            if image_script:
                print(f"Generated script with {len(image_script)} image prompts")
                for i, item in enumerate(image_script):
                    print(f"  {i+1}. {item['prompt']} - {item['duration']:.1f}s")
            else:
                print("Warning: Failed to generate image script")

            print("Step 5: Searching for images using vector search...")
            if image_script:
                # Use image prompts from the generated script
                prompts = [item["prompt"] for item in image_script]
                image_paths = self.image_manager.vector_search_images(prompts)
                print(
                    f"Found {len([p for p in image_paths if p])} valid images from {len(prompts)} prompts"
                )
            else:
                # Fallback to sentences if image script generation failed
                print("Warning: Using sentences as fallback for image search")
                image_paths = self.image_manager.vector_search_images(sentences)
                print(
                    f"Found {len([p for p in image_paths if p])} valid images from sentences"
                )

            print("Step 6: Creating video with FFmpeg...")
            print("image_script: ", image_script)
            print("image_paths: ", image_paths)
            # Choose video creation method based on blur background setting
            if use_blur_background:
                print("🌫️  Using blur background with repeating pattern...")
                # if image_script and len(image_paths) == len(image_script):
                final_video = self.create_video_with_blur_background(
                    image_paths,
                    image_script,
                    audio_path,
                    srt_path,
                    output_path,
                    audio_duration,
                    blur_strength,
                    aspect_ratio,
                )
            else:
                final_video = self.create_video_with_image_script(
                    image_paths,
                    image_script,
                    audio_path,
                    srt_path,
                    output_path,
                    audio_duration,
                    aspect_ratio,
                )

            print(f"Video generation completed: {final_video}")
            return final_video

        except Exception as e:
            print(f"Error during video generation: {str(e)}")
            raise
        finally:
            # Cleanup temp files if needed
            pass

    def cleanup(self):
        """Clean up temporary files"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def main():
    """Example usage"""
    # Sample article

    # Initialize video generator (will use environment variables for Pinecone)
    generator = VideoGenerator()

    try:
        article = ""
        web_link = "https://www.taaze.tw/products/11101053863.html"
        use_gpt_transcript = True
        custom_transcript = "想要邊賺邊存，越存越爽嗎？《上癮式存錢》這本書將刷新你對金錢的概念！它教你如何在有限的工作時間裡累積財富，如何善用投資工具穩健成長，甚至面對人生黑天鵝事件，也能保持財務穩定。從存錢意識到消費智慧，從副業收入到複利魔法，它為每個人量身訂製理財計畫，讓你的存款像雪球越滾越大。別輕視每一塊錢的力量，今天行動，讓它成為你的財富加速器！現在特價75折，快來擁有自己的財務自由藍圖吧！"
        custom_audio_path = "demo/20250814_163539_a857f8dd.wav"

        if custom_transcript:
            transcript = custom_transcript
        else:
            transcript = generator.get_transcript(article, web_link, use_gpt_transcript)
        print("Step 1: Splitting article into sentences...")
        sentences = generator.text_manager.split_article_into_sentences(transcript)
        print(f"transcript: {transcript}")

        print("Step 2: Generating audio from text...")

        if custom_audio_path:
            audio_path, audio_duration = generator.generate_audio(
                custom_audio_path=custom_audio_path
            )
        else:
            audio_path, audio_duration = generator.generate_audio(
                text_for_generation=transcript
            )
        print(f"audio_path: {audio_path}")
        print(f"audio_duration: {audio_duration}")

        print("Step 3: Generating SRT file...")
        srt_path = generator.srt_manager.generate_srt_file(sentences, audio_duration, "horizontal")
        print(f"srt_path: {srt_path}")

        print("Step 4: Generating image list script with prompts...")
        art_style = generator.image_manager.select_art_style(transcript)
        image_script = generator.image_manager.get_image_list_script(transcript, audio_duration, art_style)
        print(f"image_script: {image_script}")

        # print("Step 5: Get image paths from local...") # TODO: Need to wait for image generation to complete 
        image_paths = [
          "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\test_prompt\\test_prompt_2025-08-15_00-22-21-0001.png",
          "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\test_prompt\\test_prompt_2025-08-15_00-23-01-0001.png",
          "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\test_prompt\\test_prompt_2025-08-15_00-23-40-0001.png",
          "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\test_prompt\\test_prompt_2025-08-15_00-24-20-0001.png",
          "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\test_prompt\\test_prompt_2025-08-15_00-25-00-0001.png",
          "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\test_prompt\\test_prompt_2025-08-15_00-25-42-0001.png",
          "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\test_prompt\\test_prompt_2025-08-15_00-26-34-0001.png",
          "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\test_prompt\\test_prompt_2025-08-15_00-27-16-0001.png",
          "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\test_prompt\\test_prompt_2025-08-15_00-27-56-0001.png",
          "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\test_prompt\\test_prompt_2025-08-15_00-28-46-0001.png"
        ]

        final_video = generator.create_video_with_blur_background(
            image_paths,
            image_script,
            audio_path,
            srt_path,
            output_path="output_video.mp4",
            audio_duration=audio_duration,
            blur_strength=24,
            aspect_ratio="horizontal",
        )
        print(f"Video generation completed: {final_video}")
    except Exception as e:
        print(f"Error: {e}")
    # finally:
    #     generator.cleanup()


if __name__ == "__main__":
    main()
