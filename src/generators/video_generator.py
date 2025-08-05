#!/usr/bin/env python3
"""
FFmpeg Video Generator Script
Generates video from article text with SRT subtitles, audio, and vector-searched images
"""

import os
import sys
import re
import subprocess
import tempfile
import requests
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import librosa
from langchain_openai import AzureChatOpenAI    
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.database.pinecone_handler import PineconeHandler

# Load environment variables
load_dotenv()

class VideoGenerator:
    def __init__(self):
        """Initialize the video generator"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.temp_dir = tempfile.mkdtemp()
        self.pinecone_handler = PineconeHandler()
        
    def generate_transcript_from_article(self, article: str) -> str:
        """Generate a 1-minute transcript from an article using GPT"""
        try:
            # Get Azure OpenAI credentials from environment
            api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            
            if not api_base or not api_key:
                print("Warning: Azure OpenAI credentials not found. Returning original article.")
                return article
            
            # Initialize Azure OpenAI
            llm = AzureChatOpenAI(
                azure_endpoint=api_base,
                api_key=api_key,
                azure_deployment="gpt-4o-testing",
                api_version="2025-01-01-preview",
                temperature=0.7,
                max_tokens=500,
                timeout=None,
                max_retries=2,
            )
            
            # Create prompt template for transcript generation
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a professional video speech writer. Based on the provided article, directly give me a speech draft of about 1 minute in length. The content should be concise, engaging, and suitable for spoken video narration. Keep the original language of the article (Chinese or English); do not translate."),
                ("human", "Based on the following article, directly give me a 1-minute video speech draft:\n\n{article}. Only provide the speech draft, do not include any other explanations.")
            ])

            # Generate transcript
            chain = prompt_template | llm | StrOutputParser()
            transcript = chain.invoke({"article": article})
            
            print("✓ Successfully generated transcript using GPT")
            return transcript
            
        except Exception as e:
            print(f"Error generating transcript: {e}")
            print("Falling back to original article")
            return article
    
    def split_article_into_sentences(self, article: str) -> List[str]:
        """Split article into sentences while protecting numbers"""
        # Clean up the article
        article = re.sub(r'\s+', ' ', article.strip())
        
        # First, protect number patterns by replacing them with placeholders
        number_patterns = {}
        placeholder_counter = 0
        
        # Find and replace number patterns
        patterns_to_protect = [
            r'[-+]?\d*[.,]\d+%?',  # Decimals like -2.35%, 241.70%
            r'\(\s*[-+]?\d*[.,]?\d*%?\s*\)',  # Numbers in parentheses like (-2.35%)
            r'\$\d+[.,]?\d*',  # Currency like $100.50
            r'\d+[.,]\d*%?',  # Regular numbers with decimals
        ]
        
        protected_article = article
        for pattern in patterns_to_protect:
            matches = re.finditer(pattern, protected_article)
            for match in reversed(list(matches)):  # Reverse to maintain indices
                placeholder = f"__NUM_PLACEHOLDER_{placeholder_counter}__"
                number_patterns[placeholder] = match.group()
                protected_article = (protected_article[:match.start()] + 
                                   placeholder + 
                                   protected_article[match.end():])
                placeholder_counter += 1
        
        # Split by sentence endings, including Chinese punctuation
        sentences = re.split(r'[.!?。！？]+', protected_article)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Further split long sentences for better timing
        final_sentences = []
        for sentence in sentences:
            if len(sentence) > 80:  # If sentence is too long
                # Split by natural pause points while preserving meaning
                parts = re.split(r'(,|，|;|；|—|–|-(?=\s)|(?<=\s)while\s|(?<=\s)but\s|(?<=\s)and\s|(?<=\s)however\s)', sentence)
                
                current_part = ""
                for part in parts:
                    if part.strip() in [',', '，', ';', '；', '—', '–', '-']:
                        # Keep punctuation with previous part
                        current_part += part
                        if len(current_part) > 60:  # Break here if getting long
                            final_sentences.append(current_part.strip())
                            current_part = ""
                    elif part.strip() in ['while', 'but', 'and', 'however']:
                        # Break before conjunctions
                        if current_part.strip():
                            final_sentences.append(current_part.strip())
                        current_part = part
                    else:
                        current_part += part
                        if len(current_part) > 80:  # Break if too long
                            final_sentences.append(current_part.strip())
                            current_part = ""
                
                if current_part.strip():
                    final_sentences.append(current_part.strip())
            else:
                final_sentences.append(sentence)
        
        sentences = final_sentences
        
        # Restore the protected numbers
        restored_sentences = []
        for sentence in sentences:
            restored_sentence = sentence
            for placeholder, original_number in number_patterns.items():
                restored_sentence = restored_sentence.replace(placeholder, original_number)
            restored_sentences.append(restored_sentence)
        
        return restored_sentences
    
    def _calculate_text_width(self, text: str) -> int:
        """Calculate display width of text considering Chinese and English characters"""
        width = 0
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # Chinese characters
                width += 2
            else:  # English and other characters
                width += 1
        return width
    
    def _break_long_lines(self, text: str, max_width: int = 36) -> List[str]:
        """Break long text into multiple lines based on character width with smart chunking"""
        # Use simpler approach - split only on spaces, but protect number patterns
        words = self._smart_split_words(text)
        
        if not words:
            return [text]
        
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            
            if self._calculate_text_width(test_line) <= max_width:
                current_line = test_line
            else:
                # Save current line if it has content
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Handle very long single words/numbers - but be very careful with numbers
                    if self._is_number_pattern(word):
                        # Never break numbers, just put them on their own line
                        lines.append(word)
                        current_line = ""
                    elif self._calculate_text_width(word) > max_width:
                        # Only break non-number words
                        broken_parts = self._break_long_word(word, max_width)
                        lines.extend(broken_parts[:-1])
                        current_line = broken_parts[-1] if broken_parts else ""
                    else:
                        current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        return lines if lines else [text]
    
    def _is_number_pattern(self, text: str) -> bool:
        """Check if text contains number patterns that should never be broken"""
        number_patterns = [
            r'[-+]?\d*[.,]\d+%?',  # Decimals like -2.35%, 241.70%
            r'[-+]?\d+%?',         # Integers with optional %
            r'\$\d+[.,]?\d*',      # Currency
            r'\(\d*[.,]?\d*%?\)',  # Numbers in parentheses
        ]
        
        for pattern in number_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _smart_split_words(self, text: str) -> List[str]:
        """Split text into words while preserving numbers and special tokens"""
        # More comprehensive regex to keep numbers, percentages, and special formats together
        patterns = [
            r'[-+]?\d*[.,]\d+%?',  # Decimals like -2.35%, 241.70%, 16.48
            r'[-+]?\d+%?',         # Integers like +5%, -10, 100
            r'\$\d+[.,]?\d*',      # Currency like $100.50
            r'\(\d*[.,]?\d*%?\)',  # Numbers in parentheses like (-2.35%)
            r'\w+',                # Regular words
            r'[^\w\s]'             # Punctuation
        ]
        
        # Combine all patterns
        combined_pattern = '|'.join(f'({pattern})' for pattern in patterns)
        tokens = re.findall(combined_pattern, text)
        
        # Flatten the tuple results and filter out empty strings
        result = []
        for token_groups in tokens:
            for token in token_groups:
                if token and token.strip():
                    result.append(token)
                    break
        
        return result
    
    def _break_long_word(self, word: str, max_width: int) -> List[str]:
        """Break a single long word more intelligently"""
        if self._calculate_text_width(word) <= max_width:
            return [word]
        
        parts = []
        current_part = ""
        
        for char in word:
            test_part = current_part + char
            if self._calculate_text_width(test_part) <= max_width:
                current_part = test_part
            else:
                if current_part:
                    parts.append(current_part)
                    current_part = char
                else:
                    # Single character exceeds width (very rare)
                    parts.append(char)
                    current_part = ""
        
        if current_part:
            parts.append(current_part)
        
        return parts
    
    def generate_srt_file(self, sentences: List[str], audio_duration: float) -> str:
        """Generate SRT subtitle file with optimized timing for fast speech"""
        srt_content = []
        subtitle_index = 1
        
        # Calculate reading speed and adjust timing
        total_chars = sum(self._calculate_text_width(sentence) for sentence in sentences)
        chars_per_second = total_chars / audio_duration if audio_duration > 0 else 20
        
        # Create subtitle chunks with better timing
        subtitle_chunks = []
        
        for sentence in sentences:
            lines = self._break_long_lines(sentence)
            
            # Calculate duration based on text length and reading speed
            sentence_chars = self._calculate_text_width(sentence)
            min_duration = 1.0  # Minimum 1.0 seconds per subtitle (reduced from 1.5)
            # Adjust for shorter sentences to allow faster switching
            if sentence_chars < 30:
                min_duration = 0.8
            elif sentence_chars < 50:
                min_duration = 1.0
            else:
                min_duration = 1.2
            
            calculated_duration = max(min_duration, sentence_chars / chars_per_second * 0.9)  # Slightly faster timing
            
            subtitle_chunks.append({
                'lines': lines,
                'duration': calculated_duration,
                'char_count': sentence_chars
            })
        
        # Distribute time more evenly based on content length
        total_calculated_duration = sum(chunk['duration'] for chunk in subtitle_chunks)
        time_scale_factor = audio_duration / total_calculated_duration if total_calculated_duration > 0 else 1
        
        current_time = 0
        
        for chunk in subtitle_chunks:
            start_time = current_time
            duration = chunk['duration'] * time_scale_factor
            end_time = start_time + duration
            
            # Ensure subtitles don't exceed audio duration
            if end_time > audio_duration:
                end_time = audio_duration
            
            start_srt = self._seconds_to_srt_time(start_time)
            end_srt = self._seconds_to_srt_time(end_time)
            
            srt_content.append(f"{subtitle_index}")
            srt_content.append(f"{start_srt} --> {end_srt}")
            
            # Join lines with \n to create proper SRT format
            # Each subtitle block should have text lines separated by \n
            subtitle_text = '\n'.join(chunk['lines'])
            srt_content.append(subtitle_text)
            srt_content.append("")
            
            subtitle_index += 1
            current_time = end_time
        
        srt_file_path = os.path.join(self.temp_dir, "subtitles.srt")
        with open(srt_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_content))
        
        return srt_file_path
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _extract_sentences_from_srt(self, srt_path: str) -> List[str]:
        """Extract text sentences from SRT file"""
        sentences = []
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by double newline to get subtitle blocks
            blocks = content.strip().split('\n\n')
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:  # Valid subtitle block has at least 3 lines
                    # Skip the index and timestamp lines, get the text
                    text_lines = lines[2:]
                    text = ' '.join(text_lines).strip()
                    if text:
                        sentences.append(text)
            
            return sentences
        except Exception as e:
            print(f"Error reading SRT file: {e}")
            return []
    
    def generate_audio_from_api(self, text: str) -> Tuple[str, float]:
        """Generate audio from text using audio API"""
        try:
            # Call the audio generation API
            api_url = "https://audio-api.ask-lens.ai/video/generate"
            response = requests.post(api_url, json={"text": text})
            
            if response.status_code == 200:
                # Save audio to temp directory
                temp_audio_path = os.path.join(self.temp_dir, "generated_audio.mp3")
                with open(temp_audio_path, 'wb') as f:
                    f.write(response.content)
                
                # Get audio duration
                duration = librosa.get_duration(path=temp_audio_path)
                print(f"✓ Generated audio from API, duration: {duration:.2f}s")
                
                return temp_audio_path, duration
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Error generating audio from API: {e}")
            raise
    
    def generate_audio(self, custom_audio_path: str = None, text_for_generation: str = None) -> Tuple[str, float]:
        """Process audio file and return path and duration"""
        
        # If custom audio path is provided, use it
        if custom_audio_path and os.path.exists(custom_audio_path):
            print(f"Using custom audio file: {custom_audio_path}")
            
            # Get audio duration using multiple methods for reliability
            duration = librosa.get_duration(path=custom_audio_path)
            print(f"Librosa detected duration: {duration:.2f}s")
            
            # Also check with ffprobe for verification
            try:
                ffprobe_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                             '-of', 'csv=p=0', custom_audio_path]
                result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
                ffprobe_duration = float(result.stdout.strip())
                print(f"FFprobe detected duration: {ffprobe_duration:.2f}s")
                
                # Use ffprobe duration if significantly different
                if abs(duration - ffprobe_duration) > 1.0:
                    print(f"Duration mismatch detected, using FFprobe duration: {ffprobe_duration:.2f}s")
                    duration = ffprobe_duration
            except Exception as e:
                print(f"FFprobe duration check failed: {e}, using librosa duration")
            
            # Convert to MP3 format to ensure compatibility
            temp_audio_path = os.path.join(self.temp_dir, "audio.mp3")
            
            # Convert audio to MP3 format with specific settings
            convert_cmd = [
                'ffmpeg', '-y',
                '-i', custom_audio_path,
                '-acodec', 'mp3',
                '-ar', '44100',
                '-ac', '2',
                '-b:a', '192k',
                temp_audio_path
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
        
        # Fallback to sample audio file
        else:
            sample_audio = "data/media/raymond_bridge.mp3"
            
            if not os.path.exists(sample_audio):
                raise FileNotFoundError(f"Sample audio file not found: {sample_audio}")
            
            print(f"Using fallback sample audio: {sample_audio}")
            # Get audio duration
            duration = librosa.get_duration(path=sample_audio)
            
            # Copy to temp directory
            temp_audio_path = os.path.join(self.temp_dir, "audio.mp3")
            subprocess.run(['cp', sample_audio, temp_audio_path], check=True)
            
            return temp_audio_path, duration
    
    def vector_search_images(self, sentences: List[str], top_k: int = 5) -> List[str]:
        """Search for images using vector similarity for each sentence with deduplication"""
        image_paths = []
        used_images = set()  # Track used images to avoid duplicates
        
        if not self.pinecone_handler:
            print("Warning: Pinecone handler not initialized, using dummy image paths")
            return [None] * len(sentences)
        
        for sentence in sentences:
            try:
                # Use PineconeHandler to search with higher top_k for more options
                results = self.pinecone_handler.query_pinecone(
                    query=sentence,
                    metadata_filter={},  # No filter for now
                    index_name="image-library",
                    namespace="lens-quant",
                    top_k=top_k  # Get more candidates
                )
                
                selected_image = None
                
                if results and len(results) > 0:
                    # Try to find an unused image from the results
                    for result in results:
                        if 'metadata' in result and 'file_path' in result['metadata']:
                            candidate_path = result['metadata']['file_path']
                            
                            # Check if this image hasn't been used yet
                            if candidate_path not in used_images:
                                selected_image = candidate_path
                                used_images.add(candidate_path)
                                print(f"Found unique image for '{sentence[:30]}...': {candidate_path}")
                                break
                    
                    # If all top results are already used, use the best match anyway
                    if selected_image is None and results:
                        fallback_result = results[0]
                        if 'metadata' in fallback_result and 'file_path' in fallback_result['metadata']:
                            selected_image = fallback_result['metadata']['file_path']
                            print(f"Using duplicate image (no unique found) for '{sentence[:30]}...': {selected_image}")
                
                if selected_image:
                    image_paths.append(selected_image)
                else:
                    print(f"Warning: No valid images found for sentence: {sentence[:50]}...")
                    image_paths.append(None)
                    
            except Exception as e:
                print(f"Error searching for sentence '{sentence[:50]}...': {e}")
                image_paths.append(None)
        
        print(f"📊 Image diversity: {len(used_images)} unique images selected for {len(sentences)} sentences")
        return image_paths
    
    def convert_windows_path_to_wsl(self, windows_path: str) -> str:
        """Convert Windows path to WSL path"""
        if windows_path and windows_path.startswith('C:'):
            # Convert C:\path\to\file to /mnt/c/path/to/file
            wsl_path = windows_path.replace('C:', '/mnt/c').replace('\\', '/')
            return wsl_path
        return windows_path
    
    def find_fallback_image(self, original_path: str, used_fallbacks: set = None) -> str:
        """Find a fallback image in the same directory when the original is not found"""
        if used_fallbacks is None:
            used_fallbacks = set()
            
        wsl_path = self.convert_windows_path_to_wsl(original_path)
        directory = os.path.dirname(wsl_path)
        
        if os.path.exists(directory):
            # Look for .png files in the same directory, avoiding already used fallbacks
            available_files = []
            for file in os.listdir(directory):
                if file.endswith('.png'):
                    fallback_path = os.path.join(directory, file)
                    if os.path.exists(fallback_path) and fallback_path not in used_fallbacks:
                        available_files.append(fallback_path)
            
            # If we found unused files, return the first one
            if available_files:
                selected_fallback = available_files[0]
                used_fallbacks.add(selected_fallback)
                print(f"🔄 Using unique fallback image: {selected_fallback}")
                return selected_fallback
            
            # If all files are used, just pick the first available file
            for file in os.listdir(directory):
                if file.endswith('.png'):
                    fallback_path = os.path.join(directory, file)
                    if os.path.exists(fallback_path):
                        print(f"🔄 Using fallback image (no unique available): {fallback_path}")
                        return fallback_path
        return None
    
    def create_video_with_ffmpeg(self, 
                                 image_paths: List[str], 
                                 audio_path: str, 
                                 srt_path: str,
                                 output_path: str,
                                 audio_duration: float) -> str:
        """Create video using FFmpeg"""
        
        # Filter out None image paths and convert Windows paths to WSL paths
        valid_images = []
        used_fallbacks = set()  # Track used fallback images
        
        for img in image_paths:
            if img:
                # Convert Windows path to WSL path
                wsl_path = self.convert_windows_path_to_wsl(img)
                if os.path.exists(wsl_path):
                    valid_images.append(wsl_path)
                    print(f"✓ Found image: {wsl_path}")
                else:
                    print(f"⚠ Original image not found: {wsl_path}")
                    # Try to find a unique fallback image in the same directory
                    fallback = self.find_fallback_image(img, used_fallbacks)
                    if fallback:
                        valid_images.append(fallback)
                        print(f"✓ Using fallback instead")
                    else:
                        print(f"✗ No fallback found for: {img}")
        
        # If no valid images found, create a text-only video
        if not valid_images:
            print("Warning: No valid images found, creating a text-only video")
            return self.create_text_only_video(audio_path, srt_path, output_path)
        
        # Create image list file for FFmpeg
        # Use the duration from audio generation step to ensure consistency
        duration_per_image = audio_duration / len(valid_images)
        print(f"📊 Using audio duration from generation step: {audio_duration:.2f}s")
        
        image_list_path = os.path.join(self.temp_dir, "images.txt")
        with open(image_list_path, 'w') as f:
            for img_path in valid_images:
                f.write(f"file '{img_path}'\n")
                f.write(f"duration {duration_per_image:.2f}\n")  # Each image shows for calculated duration
        
        print(f"📊 Video timing: {len(valid_images)} images, {duration_per_image:.1f}s per image, total {audio_duration:.1f}s")
        
        # Audio duration is already known, no need to recalculate
        
        # Create main video from images
        temp_video_path = os.path.join(self.temp_dir, "temp_video.mp4")
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', image_list_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            '-vf', f'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
            temp_video_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True)
        
        # Add subtitles
        video_with_subs = os.path.join(self.temp_dir, "video_with_subs.mp4")
        
        subtitle_cmd = [
            'ffmpeg', '-y',
            '-i', temp_video_path,
            '-vf', f"subtitles={srt_path}:force_style='FontSize=24,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2'",
            '-c:a', 'copy',
            video_with_subs
        ]
        
        subprocess.run(subtitle_cmd, check=True)
        
        # Add ending video (check local directory first)
        local_ending_video = os.path.join(os.getcwd(), "data/media/LensCover.mp4")
        windows_ending_video = r"C:\Users\x7048\Documents\VideoMaker\LensCover.mp4"
        
        ending_video = None
        if os.path.exists(local_ending_video):
            ending_video = local_ending_video
            print(f"✓ Using local ending video: {local_ending_video}")
        elif os.path.exists(windows_ending_video):
            ending_video = windows_ending_video
            print(f"✓ Using Windows ending video: {windows_ending_video}")
        else:
            print("⚠ No ending video found")
        
        if ending_video:
            # First, re-encode ending video to match main video format (25fps)
            temp_ending = os.path.join(self.temp_dir, "ending_25fps.mp4")
            
            # Convert ending video to 25fps to match main video
            convert_cmd = [
                'ffmpeg', '-y',
                '-i', ending_video,
                '-r', '25',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-ar', '44100',
                '-ac', '2',
                temp_ending
            ]
            
            subprocess.run(convert_cmd, check=True)
            
            # Now concat with matching formats
            final_video_list = os.path.join(self.temp_dir, "final_list.txt")
            with open(final_video_list, 'w') as f:
                f.write(f"file '{video_with_subs}'\n")
                f.write(f"file '{temp_ending}'\n")
            
            final_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', final_video_list,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts',
                output_path
            ]
            
            subprocess.run(final_cmd, check=True)
        else:
            # Just copy the video with subtitles as final output
            subprocess.run(['cp', video_with_subs, output_path], check=True)
        
        return output_path
    
    def create_text_only_video(self, audio_path: str, srt_path: str, output_path: str) -> str:
        """Create a text-only video with subtitles when no images are available"""
        
        # Create a simple black background video with audio and subtitles
        temp_video_path = os.path.join(self.temp_dir, "text_only_video.mp4")
        
        # Get audio duration
        duration = librosa.get_duration(path=audio_path)
        
        # Create black background video with audio
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f'color=c=black:size=1920x1080:d={duration}',
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            temp_video_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True)
        
        # Add subtitles
        video_with_subs = os.path.join(self.temp_dir, "video_with_subs.mp4")
        
        subtitle_cmd = [
            'ffmpeg', '-y',
            '-i', temp_video_path,
            '-vf', f"subtitles={srt_path}:force_style='FontSize=24,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2,Alignment=2'",
            '-c:a', 'copy',
            video_with_subs
        ]
        
        subprocess.run(subtitle_cmd, check=True)
        
        # Add ending video if exists (check local directory first)
        local_ending_video = os.path.join(os.getcwd(), "data/media/LensCover.mp4")
        windows_ending_video = r"C:\Users\x7048\Documents\VideoMaker\LensCover.mp4"
        
        ending_video = None
        if os.path.exists(local_ending_video):
            ending_video = local_ending_video
            print(f"✓ Using local ending video: {local_ending_video}")
        elif os.path.exists(windows_ending_video):
            ending_video = windows_ending_video
            print(f"✓ Using Windows ending video: {windows_ending_video}")
        else:
            print("⚠ No ending video found")
        
        if ending_video:
            # First, re-encode ending video to match main video format (25fps)
            temp_ending = os.path.join(self.temp_dir, "ending_25fps.mp4")
            
            # Convert ending video to 25fps to match main video
            convert_cmd = [
                'ffmpeg', '-y',
                '-i', ending_video,
                '-r', '25',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-ar', '44100',
                '-ac', '2',
                temp_ending
            ]
            
            subprocess.run(convert_cmd, check=True)
            
            # Now concat with matching formats
            final_video_list = os.path.join(self.temp_dir, "final_list.txt")
            with open(final_video_list, 'w') as f:
                f.write(f"file '{video_with_subs}'\n")
                f.write(f"file '{temp_ending}'\n")
            
            final_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', final_video_list,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts',
                output_path
            ]
            
            subprocess.run(final_cmd, check=True)
        else:
            # Just copy the video with subtitles as final output
            subprocess.run(['cp', video_with_subs, output_path], check=True)
        
        return output_path
    
    def generate_video_from_article(self, 
                                  article: str = None, 
                                  output_path: str = "output_video.mp4", 
                                  use_gpt_transcript: bool = False,
                                  custom_srt_path: str = None,
                                  custom_audio_path: str = None) -> str:
        """Main function to generate video from article
        
        Args:
            article: The article content (optional if custom_srt_path is provided)
            output_path: Path for the output video
            use_gpt_transcript: Whether to generate a transcript using GPT first
            custom_srt_path: Path to custom SRT file (optional)
            custom_audio_path: Path to custom audio file (optional)
        """
        try:
            # Handle SRT generation or use custom SRT
            if custom_srt_path and os.path.exists(custom_srt_path):
                print("Using custom SRT file provided by user...")
                srt_path = custom_srt_path
                # Read sentences from SRT for image search
                sentences = self._extract_sentences_from_srt(custom_srt_path)
            else:
                # Need article for default flow
                if not article:
                    raise ValueError("Article content is required when not using custom SRT")
                
                # Optional: Generate transcript from article using GPT
                if use_gpt_transcript:
                    print("Step 0: Generating transcript from article using GPT...")
                    transcript = self.generate_transcript_from_article(article)
                    print("Transcript generated successfully")
                    # Use the transcript instead of the original article
                    article = transcript
                    # Don't translate since the transcript is already in the desired language
                
                print("Step 1: Splitting article into sentences...")
                sentences = self.split_article_into_sentences(article)
                print(f"Found {len(sentences)} sentences")
            
            # Handle audio generation or use custom audio
            if custom_audio_path and os.path.exists(custom_audio_path):
                print("Using custom audio file provided by user...")
                audio_path, audio_duration = self.generate_audio(custom_audio_path=custom_audio_path)
                print(f"Custom audio duration: {audio_duration:.2f} seconds")
            else:
                print("Step 2: Generating audio from text...")
                # Use the processed article text for audio generation
                text_for_audio = article if not use_gpt_transcript else article
                audio_path, audio_duration = self.generate_audio(text_for_generation=text_for_audio)
                print(f"Audio duration: {audio_duration:.2f} seconds")
            
            # Generate SRT if not using custom
            if not custom_srt_path:
                print("Step 3: Generating SRT file...")
                srt_path = self.generate_srt_file(sentences, audio_duration)
                print(f"SRT file created: {srt_path}")
            
            print("Step 4: Searching for images using vector search...")
            image_paths = self.vector_search_images(sentences)
            print(f"Found {len([p for p in image_paths if p])} valid images")
            
            print("Step 5: Creating video with FFmpeg...")
            final_video = self.create_video_with_ffmpeg(
                image_paths, audio_path, srt_path, output_path, audio_duration
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
    sample_article = """
    量化交易是一種利用電腦程式和數學模型，
    根據大量數據自動執行金融交易的方式。
    透過設計和優化交易策略，
    量化交易能夠在極短的時間內分析市場資訊、
    發現機會並下單，減少人為情緒干擾。
    這種方法廣泛應用於股票、期貨、外匯等市場，
    幫助投資人提升效率和風險控管。
    不過，量化交易也需要嚴格的風險管理和持續的策略調整
    才能在快速變化的金融市場中保持競爭力。
    """
    fixed_article = """
    Today’s portfolio underperformance reflects a temporary misalignment between expectations and realized earnings momentum across our cyclical long positions. CCL (-2.35%) and FDX (-2.86%) underperformed broader market declines (-1.64%), underscoring their sensitivity to short-term demand fluctuations. ORLY’s modest gain (+0.81%) provided limited balance, while PNW (+0.07%), our defensive short, failed to deliver sufficient downside protection due to stagnant movement in utility valuations. This highlights the importance of refining risk-parity mechanisms to manage unhedged exposure during volatile macro conditions.
    The strategy continues to exhibit strong long-term performance metrics—241.70% annualized returns with a Sharpe ratio of 16.48—validating our systematic approach. However, today’s results emphasize that even high-probability strategies face episodic dislocations, particularly when assumption-driven trades encounter unexpected macroeconomic noise. Consistent diversification beyond cyclical sectors will be critical to reduce drawdowns and sustain our edge.
    The current market environment remains stable but fragile, with growth-oriented cyclicals facing decelerating post-recovery momentum. Continued earnings revisions and analyst expectation gaps within consumer-sensitive sectors suggest opportunity, but these must be weighed against mounting uncertainty in macro policy signals and demand elasticity. A more robust hedge across non-cyclical, defensive sectors may enhance resilience.
    Our holdings reflect a deliberate focus on earnings revision momentum and analyst expectation gaps. Long positions in CCL and FDX are based on anticipated topline recovery tied to macro reopening tailwinds, while ORLY brings strong fundamental relative valuation. PNW, our short, challenges overvalued defensive utilities, balancing exposure. These decisions are rooted in systematic valuation frameworks, though current execution reveals gaps in macro overlay calibration.
    As always, designing around the system requires constant iteration. Most risks lie in assumptions we mistakenly deem certain—a principle that today reinforces. Diversification remains the only true free lunch.
    """
    
    # Initialize video generator (will use environment variables for Pinecone)
    generator = VideoGenerator()
    
    try:
        # custom_audio = "data/media/somer_smaple.wav"
        # # Example 1: Generate audio from text using API
        # print("=== Example 1: Generate audio from text ===")
        # output_file = generator.generate_video_from_article(
        #     article=sample_article, 
        #     output_path="sample_video_generated_audio.mp4",
        #     use_gpt_transcript=False,
        #     custom_srt_path=None,
        #     custom_audio_path=custom_audio  # Will use API to generate audio
        # )
        # print(f"Success! Video with generated audio saved to: {output_file}")
        
        # Example 2: Use custom audio file (if available)
        custom_audio = "data/media/raymond_bridge8-3.mp3"
        if os.path.exists(custom_audio):
            print("\n=== Example 2: Use custom audio file ===")
            output_file_2 = generator.generate_video_from_article(
                article=fixed_article, 
                output_path="raymond_bridge.mp4",
                use_gpt_transcript=False,
                custom_srt_path=None,
                custom_audio_path=custom_audio
            )
            print(f"Success! Video with custom audio saved to: {output_file_2}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        generator.cleanup()

if __name__ == "__main__":
    main()