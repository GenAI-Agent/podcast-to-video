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
import librosa
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from elevenlabs.client import ElevenLabs

# Add project root to sys.path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.database.pinecone_handler import PineconeHandler
from src.utils.json_image_loader import JsonImageLoader

# Load environment variables
load_dotenv()
transcript_prompt = """
    You are a professional video speech writer. Based on the provided article, directly give me a speech draft of about 1 minute in length. The content should be concise, engaging, and suitable for spoken video narration.
    Keep the original language of the article (Chinese or English); do not translate.
    But when you use Chinese, you can only use Traditional Chinese.
"""
transcript_user_prompt = """
    Based on the following article, directly give me a 1-minute video speech draft:

    {article}

    Only provide the speech draft, do not include any other explanations.
    Dont include any other text or symbols in your response.
    
"""
image_prompt = """
    You are a professional video director. Based on the provided transcript and audio duration, generate a script with image descriptions and durations.
    
    Each image should:
    - Have a clear, searchable description (tags/keywords that can be used to find relevant images)
    - Have a duration of at least 1 second
    - Together, all images should cover the entire audio duration
    - Be relevant to the content being spoken at that time
    
    The descriptions should be specific and visual, suitable for image search.
"""
image_user_prompt = """
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


class VideoGenerator:
    def __init__(self, restricted_json_file: str = None, topic: str = None):
        """Initialize the video generator

        Args:
            restricted_json_file: Optional path to JSON file containing restricted image dataset.
                                If provided, image selection will be limited to images in this dataset.
            topic: Selected topic for category-based filtering (optional)
        """
        self.temp_dir = tempfile.mkdtemp()
        # self.temp_dir = "temp"
        self.pinecone_handler = PineconeHandler()
        self.topic = topic
        self.topic_categories = self._get_topic_categories()

        # Initialize restricted image loader if JSON file is provided
        self.json_image_loader = None
        self.restricted_task_ids = set()
        if restricted_json_file and os.path.exists(restricted_json_file):
            try:
                self.json_image_loader = JsonImageLoader(restricted_json_file)
                self.restricted_task_ids = self.json_image_loader.get_all_task_ids()
                
                # Filter by topic categories if specified
                if self.topic and self.topic in self.topic_categories:
                    allowed_categories = self.topic_categories[self.topic]
                    print(f"🎯 Topic '{self.topic}' allows categories: {allowed_categories}")
                    
                print(f"🔒 Restricted image mode enabled: {len(self.restricted_task_ids)} images available")
            except Exception as e:
                print(f"⚠️ Warning: Could not load restricted image dataset: {e}")
                self.json_image_loader = None

    def _get_topic_categories(self) -> dict:
        """
        Define mapping from topics to allowed categories and subcategories
        
        Returns:
            Dict mapping topic names to lists of allowed categories
        """
        return {
            'astrology': ['REAL_ESTATE', 'FINANCE', 'INVESTMENT', 'ASTROLOGY', 'ZODIAC', 'STARS'],
            'trading': ['REAL_ESTATE', 'FINANCE', 'INVESTMENT', 'BUSINESS', 'DAY_TRADING', 'TRADING', 'STOCK_MARKET'],
            'fantasy': ['FANTASY_ADVENTURE', 'FANTASY', 'MAGICAL'],
            'horror': ['SPOOKY_STORY', 'HORROR', 'DARK', 'SUPERNATURAL'],
            'romance': ['ROMANCE', 'LOVE', 'ROMANTIC'],
            'drama': ['DRAMA', 'EMOTIONAL', 'HUMAN_STORY'],
            'thriller': ['THRILLER', 'SUSPENSE', 'ACTION', 'MYSTERY']
        }

    def _collect_image_previews(self, image_paths: List[str], descriptions: List[str]) -> List[dict]:
        """
        Collect image preview information for frontend display
        
        Args:
            image_paths: List of image file paths (may contain None values)
            descriptions: List of descriptions/sentences for each image
            
        Returns:
            List of dictionaries with image preview information
        """
        previews = []
        
        for i, (path, description) in enumerate(zip(image_paths, descriptions)):
            preview_info = {
                "index": i,
                "description": description[:100] + "..." if len(description) > 100 else description,
                "has_image": path is not None,
                "image_path": None,
                "fallback_used": False,
                "error_reason": None
            }
            
            if path:
                # Convert to relative path for serving
                wsl_path = self.convert_windows_path_to_wsl(path)
                if os.path.exists(wsl_path):
                    # Create a web-accessible path relative to allowed directories
                    # API server checks these allowed directories:
                    # - /home/fluxmind/batch_image/data
                    # - /mnt/c/Users/x7048/Documents/ComfyUI/output
                    
                    # Try to create relative path from known base directories
                    allowed_bases = [
                        "/home/fluxmind/batch_image/data",
                        "/mnt/c/Users/x7048/Documents/ComfyUI/output"
                    ]
                    
                    web_path = os.path.basename(wsl_path)  # fallback to just filename
                    for base_dir in allowed_bases:
                        try:
                            if wsl_path.startswith(base_dir):
                                web_path = os.path.relpath(wsl_path, base_dir)
                                break
                        except ValueError:
                            continue
                    
                    preview_info["image_path"] = web_path
                    preview_info["full_path"] = wsl_path
                    
                    # Get additional metadata if available from JSON loader
                    if self.json_image_loader:
                        try:
                            # Find the image info by path matching
                            for category_name, category_data in self.json_image_loader.data.items():
                                for img_info in category_data:
                                    # img_info is a dict, not an object
                                    if img_info.get("file_path", "").endswith(os.path.basename(wsl_path)):
                                        preview_info["category"] = category_name
                                        preview_info["tags"] = img_info.get("tags", "")
                                        break
                        except Exception as e:
                            print(f"Warning: Could not get metadata for {wsl_path}: {e}")
                else:
                    preview_info["fallback_used"] = True
                    preview_info["error_reason"] = f"File not found: {wsl_path}" if path else "No image path returned from search"
            else:
                preview_info["error_reason"] = "No image found for this description"
            
            previews.append(preview_info)
        
        return previews

    def generate_transcript_from_article(self, article: str) -> str:
        """Generate a 1-minute transcript from an article using GPT"""
        try:
            # Get Azure OpenAI credentials from environment
            api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")

            if not api_base or not api_key:
                print(
                    "Warning: Azure OpenAI credentials not found. Returning original article."
                )
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
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        transcript_prompt,
                    ),
                    (
                        "human",
                        transcript_user_prompt,
                    ),
                ]
            )

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
        """Split article into short sentences (max 10 chars) without punctuation"""
        # Clean up the article and replace commas with spaces
        article = article.strip()
        article = re.sub(r"[,，]", " ", article)  # Replace commas with spaces
        article = re.sub(r"\s+", " ", article)  # Clean up multiple spaces

        # Split by sentence endings, including Chinese punctuation
        sentences = re.split(r"[.!?。！？]+", article)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Further split sentences to ensure max 25 characters
        final_sentences = []
        max_chars = 25  # Maximum 25 characters per sentence

        for sentence in sentences:
            # Keep the sentence as is initially, don't strip punctuation yet
            sentence = sentence.strip()

            if len(sentence) <= max_chars:
                if sentence:  # Only add non-empty sentences
                    final_sentences.append(sentence)
            else:
                # Split long sentences into chunks of max 25 characters
                # Try to split at natural word boundaries while preserving paired punctuation
                words = sentence.split()
                current_chunk = ""

                for word in words:
                    # Check if adding this word would exceed limit
                    test_chunk = current_chunk + (" " if current_chunk else "") + word

                    if len(test_chunk) <= max_chars:
                        current_chunk = test_chunk
                    else:
                        # Save current chunk if not empty and meaningful
                        if current_chunk and len(current_chunk.strip()) > 0:
                            final_sentences.append(current_chunk)

                        # Start new chunk with current word
                        if len(word) > max_chars:
                            # If single word is too long, check if it contains paired punctuation
                            # Keep book titles and parentheses together
                            if any(
                                p in word for p in ["《》", "「」", "（）", "()", "“”"]
                            ):
                                final_sentences.append(
                                    word
                                )  # Keep paired punctuation together
                            else:
                                for i in range(0, len(word), max_chars):
                                    chunk = word[i : i + max_chars]
                                    if chunk and chunk.strip():
                                        final_sentences.append(chunk)
                            current_chunk = ""
                        else:
                            current_chunk = word

                # Add remaining chunk
                if current_chunk and current_chunk.strip():
                    final_sentences.append(current_chunk)

        # Final cleanup: merge standalone punctuation with adjacent sentences
        cleaned_sentences = []
        for i, sentence in enumerate(final_sentences):
            sentence = sentence.strip()
            # Check if this is just standalone punctuation
            content_without_punct = sentence.strip(",.，。、；;：:》」）)")

            if len(content_without_punct) == 0:  # Just punctuation
                # Try to merge with previous sentence if possible
                if (
                    cleaned_sentences
                    and len(cleaned_sentences[-1]) + len(sentence) <= max_chars
                ):
                    cleaned_sentences[-1] += sentence
                # Otherwise skip standalone punctuation
            elif len(sentence) <= 2 and i > 0:  # Very short sentence, try to merge
                if len(cleaned_sentences[-1]) + len(sentence) + 1 <= max_chars:
                    cleaned_sentences[-1] += sentence
                else:
                    cleaned_sentences.append(sentence)
            else:
                if sentence:  # Only add non-empty sentences
                    cleaned_sentences.append(sentence)

        return cleaned_sentences

    def extract_web_content(self, web_link: str) -> str:
        """Extract text content from a web page

        Args:
            web_link: URL of the web page to extract content from

        Returns:
            str: Extracted text content from the web page
        """
        try:
            # Validate URL
            parsed_url = urlparse(web_link)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL: {web_link}")

            # Set headers to mimic a browser request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            # Fetch the web page
            print(f"Fetching content from: {web_link}")
            response = requests.get(web_link, headers=headers, timeout=30)
            response.raise_for_status()

            # Parse HTML content
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text from common content elements
            content_elements = soup.find_all(
                ["p", "h1", "h2", "h3", "h4", "h5", "h6", "article", "section", "main"]
            )

            # If no specific content elements found, get all text
            if not content_elements:
                text = soup.get_text()
            else:
                text_parts = []
                for element in content_elements:
                    element_text = element.get_text().strip()
                    if element_text:
                        text_parts.append(element_text)
                text = "\n\n".join(text_parts)

            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Ensure we have meaningful content
            if len(text) < 100:
                raise ValueError(
                    "Extracted content is too short. The page might be dynamic or protected."
                )

            print(f"✓ Successfully extracted {len(text)} characters from web page")
            return text

        except requests.RequestException as e:
            print(f"Error fetching web page: {e}")
            raise
        except Exception as e:
            print(f"Error extracting web content: {e}")
            raise

    def get_image_list_script(
        self, transcript: str, audio_duration: float
    ) -> List[dict]:
        """Generate a list of image descriptions with durations for video creation using GPT

        Args:
            transcript: The transcript content
            audio_duration: Total duration of the audio in seconds

        Returns:
            List[dict]: A list of dictionaries with format [{description:"tags...",duration:1.2},...]
        """
        try:
            # Get Azure OpenAI credentials from environment
            api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")

            if not api_base or not api_key:
                print(
                    "Warning: Azure OpenAI credentials not found. Returning empty list."
                )
                return []

            # Initialize Azure OpenAI
            llm = AzureChatOpenAI(
                azure_endpoint=api_base,
                api_key=api_key,
                azure_deployment="gpt-4o-testing",
                api_version="2025-01-01-preview",
                temperature=0.7,
                max_tokens=2000,
                timeout=None,
                max_retries=2,
            )

            # System prompt placeholder - can be customized later
            system_prompt = image_prompt

            # Create prompt template for image script generation
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    (
                        "human",
                        image_user_prompt,
                    ),
                ]
            )

            # Generate image script
            chain = prompt_template | llm | StrOutputParser()
            response = chain.invoke(
                {"transcript": transcript, "audio_duration": audio_duration}
            )

            # Parse the JSON response
            import json

            try:
                # Extract JSON array from the response
                # Sometimes GPT might add extra text, so we need to find the JSON part
                json_start = response.find("[")
                json_end = response.rfind("]") + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    image_list = json.loads(json_str)
                else:
                    raise ValueError("No JSON array found in response")

                # Validate and ensure minimum duration
                validated_list = []
                for item in image_list:
                    if (
                        isinstance(item, dict)
                        and "description" in item
                        and "duration" in item
                    ):
                        # Ensure minimum duration of 1 second
                        duration = max(1.0, float(item["duration"]))
                        validated_list.append(
                            {
                                "description": str(item["description"]),
                                "duration": duration,
                            }
                        )

                # Adjust durations to match audio duration exactly
                total_duration = sum(item["duration"] for item in validated_list)
                if total_duration > 0 and abs(total_duration - audio_duration) > 0.1:
                    # Scale durations proportionally
                    scale_factor = audio_duration / total_duration
                    for item in validated_list:
                        item["duration"] = max(1.0, item["duration"] * scale_factor)

                print(
                    f"✓ Successfully generated {len(validated_list)} image descriptions with durations"
                )
                return validated_list

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Response was: {response}")
                return []

        except Exception as e:
            print(f"Error generating image script: {e}")
            return []

    def _calculate_text_width(self, text: str) -> int:
        """Calculate display width of text considering Chinese and English characters"""
        width = 0
        for char in text:
            if "\u4e00" <= char <= "\u9fff":  # Chinese characters
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
            r"[-+]?\d*[.,]\d+%?",  # Decimals like -2.35%, 241.70%
            r"[-+]?\d+%?",  # Integers with optional %
            r"\$\d+[.,]?\d*",  # Currency
            r"\(\d*[.,]?\d*%?\)",  # Numbers in parentheses
        ]

        for pattern in number_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _smart_split_words(self, text: str) -> List[str]:
        """Split text into words while preserving numbers and special tokens"""
        # More comprehensive regex to keep numbers, percentages, and special formats together
        patterns = [
            r"[-+]?\d*[.,]\d+%?",  # Decimals like -2.35%, 241.70%, 16.48
            r"[-+]?\d+%?",  # Integers like +5%, -10, 100
            r"\$\d+[.,]?\d*",  # Currency like $100.50
            r"\(\d*[.,]?\d*%?\)",  # Numbers in parentheses like (-2.35%)
            r"\w+",  # Regular words
            r"[^\w\s]",  # Punctuation
        ]

        # Combine all patterns
        combined_pattern = "|".join(f"({pattern})" for pattern in patterns)
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
        total_chars = sum(
            self._calculate_text_width(sentence) for sentence in sentences
        )
        chars_per_second = total_chars / audio_duration if audio_duration > 0 else 20

        # Create subtitle chunks with better timing
        subtitle_chunks = []

        for sentence in sentences:
            lines = self._break_long_lines(sentence)

            # Calculate duration based on text length and reading speed
            sentence_chars = self._calculate_text_width(sentence)
            # Increased minimum durations for better readability
            min_duration = 1.5  # Minimum 1.5 seconds per subtitle for better readability
            # Adjust for shorter sentences to allow appropriate reading time
            if sentence_chars < 30:
                min_duration = 1.2  # Increased from 0.8
            elif sentence_chars < 50:
                min_duration = 1.5  # Increased from 1.0
            else:
                min_duration = 2.0  # Increased from 1.2

            calculated_duration = max(
                min_duration, sentence_chars / chars_per_second * 1.1
            )  # Slightly slower timing for better reading comprehension

            subtitle_chunks.append(
                {
                    "lines": lines,
                    "duration": calculated_duration,
                    "char_count": sentence_chars,
                }
            )

        # Distribute time more evenly based on content length
        total_calculated_duration = sum(chunk["duration"] for chunk in subtitle_chunks)
        time_scale_factor = (
            audio_duration / total_calculated_duration
            if total_calculated_duration > 0
            else 1
        )

        current_time = 0

        for chunk in subtitle_chunks:
            start_time = current_time
            duration = chunk["duration"] * time_scale_factor
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
            subtitle_text = "\n".join(chunk["lines"])
            srt_content.append(subtitle_text)
            srt_content.append("")

            subtitle_index += 1
            current_time = end_time

        srt_file_path = os.path.join(self.temp_dir, "subtitles.srt")
        with open(srt_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))

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
            with open(srt_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Split by double newline to get subtitle blocks
            blocks = content.strip().split("\n\n")

            for block in blocks:
                lines = block.strip().split("\n")
                if len(lines) >= 3:  # Valid subtitle block has at least 3 lines
                    # Skip the index and timestamp lines, get the text
                    text_lines = lines[2:]
                    text = " ".join(text_lines).strip()
                    if text:
                        sentences.append(text)

            return sentences
        except Exception as e:
            print(f"Error reading SRT file: {e}")
            return []

    def generate_audio_from_api(self, text: str) -> Tuple[str, float]:
        """Generate audio from text using ElevenLabs API with text chunking for long texts"""
        try:
            # Get ElevenLabs API credentials from environment
            elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
            if not elevenlabs_api_key:
                raise ValueError("ELEVENLABS_API_KEY environment variable not set")

            # Initialize ElevenLabs client
            elevenlabs = ElevenLabs(api_key=elevenlabs_api_key)

            # Voice configuration - using JBFqnCBsd6RMkjVDRZzb as in your example
            voice_id = "JBFqnCBsd6RMkjVDRZzb"
            # voice_id = "fQj4gJSexpu8RDE2Ii5m"

            print(
                f"Generating audio with ElevenLabs API (text length: {len(text)} chars)..."
            )

            # ElevenLabs has a 10,000 character limit, so we need to chunk long texts
            MAX_CHARS = 9500  # Leave some buffer for safety
            
            if len(text) <= MAX_CHARS:
                # Single request for short text
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
            else:
                # Multiple requests for long text - chunk by sentences
                print(f"Text too long ({len(text)} chars), splitting into chunks...")
                
                # Split text into chunks by sentences while respecting character limit
                chunks = self._split_text_into_chunks(text, MAX_CHARS)
                print(f"Split text into {len(chunks)} chunks")
                
                # Generate audio for each chunk
                audio_files = []
                total_duration = 0
                
                for i, chunk in enumerate(chunks):
                    print(f"Generating audio for chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
                    
                    audio = elevenlabs.text_to_speech.convert(
                        text=chunk,
                        voice_id=voice_id,
                        model_id="eleven_multilingual_v2",
                        output_format="mp3_44100_128",
                    )

                    # Save chunk audio to temp file
                    chunk_audio_path = os.path.join(self.temp_dir, f"audio_chunk_{i}.mp3")
                    with open(chunk_audio_path, "wb") as f:
                        for audio_chunk in audio:
                            f.write(audio_chunk)
                    
                    audio_files.append(chunk_audio_path)
                    chunk_duration = librosa.get_duration(path=chunk_audio_path)
                    total_duration += chunk_duration
                    print(f"✓ Chunk {i+1} generated, duration: {chunk_duration:.2f}s")
                
                # Concatenate all audio files
                final_audio_path = os.path.join(self.temp_dir, "generated_audio.mp3")
                self._concatenate_audio_files(audio_files, final_audio_path)
                
                # Clean up chunk files
                for chunk_file in audio_files:
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
                
                print(f"✓ Generated complete audio from {len(chunks)} chunks, total duration: {total_duration:.2f}s")
                return final_audio_path, total_duration

        except Exception as e:
            print(f"Error generating audio from ElevenLabs API: {e}")
            raise

    def _split_text_into_chunks(self, text: str, max_chars: int) -> List[str]:
        """Split text into chunks that respect sentence boundaries and character limits"""
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed limit, start new chunk
            if len(current_chunk) + len(sentence) + 1 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence too long - split by words
                    if len(sentence) > max_chars:
                        words = sentence.split()
                        word_chunk = ""
                        for word in words:
                            if len(word_chunk) + len(word) + 1 > max_chars:
                                if word_chunk:
                                    chunks.append(word_chunk.strip())
                                    word_chunk = word
                                else:
                                    # Single word too long - just truncate
                                    chunks.append(word[:max_chars])
                            else:
                                word_chunk += " " + word if word_chunk else word
                        if word_chunk:
                            current_chunk = word_chunk
                    else:
                        current_chunk = sentence
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _concatenate_audio_files(self, audio_files: List[str], output_path: str):
        """Concatenate multiple audio files into a single file using ffmpeg"""
        if not audio_files:
            raise ValueError("No audio files to concatenate")
        
        if len(audio_files) == 1:
            # Just copy the single file
            import shutil
            shutil.copy2(audio_files[0], output_path)
            return
        
        # Create a temporary file list for ffmpeg concat
        concat_file_path = os.path.join(self.temp_dir, "audio_concat_list.txt")
        with open(concat_file_path, 'w') as f:
            for audio_file in audio_files:
                f.write(f"file '{os.path.abspath(audio_file)}'\n")
        
        # Use ffmpeg to concatenate
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file_path,
            "-c", "copy",
            output_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg concatenation failed: {result.stderr}")
        
        # Clean up temp file
        if os.path.exists(concat_file_path):
            os.remove(concat_file_path)

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

    def vector_search_images(
        self, descriptions: List[str], top_k: int = 10
    ) -> List[str]:
        """Search for images using vector similarity for each sentence with deduplication"""
        image_paths = []
        used_images = set()  # Track used images to avoid duplicates

        # Show topic filtering status
        if self.topic and self.topic in self.topic_categories:
            print(f"🎯 Topic-based filtering active: '{self.topic}' -> categories {self.topic_categories[self.topic]}")
        elif self.json_image_loader:
            print(f"🔒 Restricted dataset mode (no topic filter)")
        else:
            print(f"🌍 Full dataset mode (no restrictions)")

        if not self.pinecone_handler:
            print("Warning: Pinecone handler not initialized, using dummy image paths")
            return [None] * len(descriptions)

        for description in descriptions:
            try:
                # Use PineconeHandler to search with higher top_k for more options
                results = self.pinecone_handler.query_pinecone(
                    query=description,
                    metadata_filter={},  # No filter for now
                    index_name="image-library",
                    namespace="description",
                    top_k=top_k,  # Get more candidates
                )

                selected_image = None

                if results and len(results) > 0:
                    # Try to find an unused image from the results that actually exists
                    for result in results:
                        if "metadata" in result and "file_path" in result["metadata"]:
                            candidate_path = result["metadata"]["file_path"]

                            # Check if image is allowed in restricted mode
                            if self.json_image_loader and not self._is_image_allowed(result):
                                continue

                            # Validate that the file actually exists
                            wsl_candidate_path = self.convert_windows_path_to_wsl(candidate_path)
                            if not os.path.exists(wsl_candidate_path):
                                print(f"⚠ Skipping missing file: {candidate_path}")
                                continue

                            # Check if this image hasn't been used yet
                            if candidate_path not in used_images:
                                selected_image = candidate_path
                                used_images.add(candidate_path)
                                restriction_note = " (restricted)" if self.json_image_loader else ""
                                print(
                                    f"Found unique image{restriction_note} for '{description[:30]}...': {candidate_path}"
                                )
                                break

                    # If all top results are already used, try to find any allowed image as fallback
                    if selected_image is None and results:
                        for result in results:
                            if "metadata" in result and "file_path" in result["metadata"]:
                                candidate_path = result["metadata"]["file_path"]
                                
                                # Check if image is allowed in restricted mode
                                if self.json_image_loader and not self._is_image_allowed(result):
                                    continue

                                # Validate that the file actually exists
                                wsl_candidate_path = self.convert_windows_path_to_wsl(candidate_path)
                                if not os.path.exists(wsl_candidate_path):
                                    continue

                                selected_image = candidate_path
                                restriction_note = " (restricted)" if self.json_image_loader else ""
                                print(
                                    f"Using duplicate image{restriction_note} (no unique found) for '{description[:30]}...': {selected_image}"
                                )
                                break

                if selected_image:
                    image_paths.append(selected_image)
                else:
                    # If in restricted mode and no images found, use smart fallback strategy
                    if self.json_image_loader and results:
                        if self.topic and self.topic in self.topic_categories:
                            # When topic is specified, try fallback within topic first
                            fallback_used = False
                            for result in results[:3]:  # Try top 3 results
                                if "metadata" in result and "file_path" in result["metadata"]:
                                    candidate_path = result["metadata"]["file_path"]
                                    
                                    # Validate that the file actually exists
                                    wsl_candidate_path = self.convert_windows_path_to_wsl(candidate_path)
                                    if not os.path.exists(wsl_candidate_path):
                                        continue
                                    
                                    selected_image = candidate_path
                                    image_paths.append(selected_image)
                                    print(f"🔄 Using topic fallback for '{description[:30]}...': {selected_image}")
                                    fallback_used = True
                                    break
                            
                            if not fallback_used:
                                print(f"🚫 No images found for '{description[:30]}...' in topic '{self.topic}'")
                                image_paths.append(None)
                        else:
                            # Only use fallback if no topic restriction is active
                            print(f"🔄 No restricted images found for '{description[:30]}...', using fallback")
                            fallback_found = False
                            for result in results[:5]:  # Try top 5 results for fallback
                                if "metadata" in result and "file_path" in result["metadata"]:
                                    candidate_path = result["metadata"]["file_path"]
                                    
                                    # Validate that the file actually exists
                                    wsl_candidate_path = self.convert_windows_path_to_wsl(candidate_path)
                                    if not os.path.exists(wsl_candidate_path):
                                        continue
                                    
                                    selected_image = candidate_path
                                    image_paths.append(selected_image)
                                    print(f"   Using fallback image: {selected_image}")
                                    fallback_found = True
                                    break
                            
                            if not fallback_found:
                                image_paths.append(None)
                    else:
                        print(
                            f"⚠ No search results for description: {description[:50]}..."
                        )
                        image_paths.append(None)

            except Exception as e:
                print(f"Error searching for description '{description[:50]}...': {e}")
                image_paths.append(None)
        
        # Final attempt: if we have NO images at all, try to find at least one working image
        if all(path is None for path in image_paths) and len(descriptions) > 0:
            print("🔥 Emergency fallback: No images found, searching for any available image...")
            try:
                # Use a very general search term to find any image
                emergency_results = self.pinecone_handler.query_pinecone(
                    query="image picture photo",
                    metadata_filter={},
                    index_name="image-library", 
                    namespace="description",
                    top_k=20
                )
                
                for result in emergency_results:
                    if "metadata" in result and "file_path" in result["metadata"]:
                        candidate_path = result["metadata"]["file_path"]
                        wsl_candidate_path = self.convert_windows_path_to_wsl(candidate_path)
                        if os.path.exists(wsl_candidate_path):
                            # Use this image for the first description
                            image_paths[0] = candidate_path
                            print(f"🆘 Emergency fallback image found: {candidate_path}")
                            break
            except Exception as e:
                print(f"Emergency fallback failed: {e}")

        # Report detailed image assignment statistics
        found_count = len([p for p in image_paths if p is not None])
        missing_count = len(descriptions) - found_count
        print(
            f"📊 Image diversity: {len(used_images)} unique images selected for {len(descriptions)} descriptions"
        )
        print(f"📊 Assignment result: {found_count}/{len(descriptions)} images found, {missing_count} missing")
        
        # Log any missing assignments for debugging
        if missing_count > 0:
            for i, (path, desc) in enumerate(zip(image_paths, descriptions)):
                if path is None:
                    print(f"❌ Missing image #{i+1}: '{desc[:50]}...'")
        
        return image_paths

    def _is_image_allowed(self, search_result: dict) -> bool:
        """
        Check if an image from search results is allowed in restricted mode
        Enforces both dataset restrictions and topic category filtering

        Args:
            search_result: Pinecone search result with metadata

        Returns:
            True if image is allowed, False otherwise
        """
        if not self.json_image_loader:
            return True  # No restrictions if no JSON loader

        # Try to extract task_id from metadata
        metadata = search_result.get("metadata", {})
        found_image_info = None

        # Check various possible fields that might contain task_id
        possible_task_id_fields = ["task_id", "id", "uuid", "image_id"]

        for field in possible_task_id_fields:
            if field in metadata:
                task_id = metadata[field]
                if task_id in self.restricted_task_ids:
                    found_image_info = self.json_image_loader.get_image_by_task_id(task_id)
                    break

        # If no task_id match found, check file path or name matching
        if not found_image_info:
            file_path = metadata.get("file_path", "")
            if file_path:
                # Extract filename from path
                filename = os.path.basename(file_path)

                # Check if any image in our dataset has a matching filename pattern
                for image_info in self.json_image_loader.images:
                    if image_info.file_name and image_info.file_name in filename:
                        found_image_info = image_info
                        break

        if not found_image_info:
            return False  # Not found in restricted dataset

        # If we have a topic restriction, check category matching
        if self.topic and self.topic in self.topic_categories:
            allowed_categories = self.topic_categories[self.topic]
            image_category = found_image_info.category

            # Check if the image's category is in the allowed list for this topic
            if image_category not in allowed_categories:
                print(f"🚫 Filtered out image from category '{image_category}' (topic '{self.topic}' allows: {allowed_categories})")
                return False
            else:
                # Image matches topic category - show success
                print(f"✅ Image from category '{image_category}' matches topic '{self.topic}'")

        return True  # Image passes all restrictions

    def convert_windows_path_to_wsl(self, windows_path: str) -> str:
        """Convert Windows path to WSL path"""
        if windows_path and windows_path.startswith("C:"):
            # Convert C:\path\to\file to /mnt/c/path/to/file
            wsl_path = windows_path.replace("C:", "/mnt/c").replace("\\", "/")
            return wsl_path
        return windows_path

    def find_fallback_image(
        self, original_path: str, used_fallbacks: set = None
    ) -> str:
        """Find a fallback image in the same directory or use project fallbacks when the original is not found"""
        if used_fallbacks is None:
            used_fallbacks = set()

        wsl_path = self.convert_windows_path_to_wsl(original_path)
        directory = os.path.dirname(wsl_path)

        if os.path.exists(directory):
            # Look for .png files in the same directory, avoiding already used fallbacks
            available_files = []
            for file in os.listdir(directory):
                if file.endswith((".png", ".jpg")):
                    fallback_path = os.path.join(directory, file)
                    if (
                        os.path.exists(fallback_path)
                        and fallback_path not in used_fallbacks
                    ):
                        available_files.append(fallback_path)

            # If we found unused files, return the first one
            if available_files:
                selected_fallback = available_files[0]
                used_fallbacks.add(selected_fallback)
                print(f"🔄 Using unique fallback image: {selected_fallback}")
                return selected_fallback

            # If all files are used, just pick the first available file
            for file in os.listdir(directory):
                if file.endswith((".png", ".jpg")):
                    fallback_path = os.path.join(directory, file)
                    if os.path.exists(fallback_path):
                        print(
                            f"🔄 Using fallback image (no unique available): {fallback_path}"
                        )
                        return fallback_path
        
        # If original directory doesn't exist, use project fallback images
        fallback_dir = os.path.join(os.getcwd(), "fallback_images")
        if os.path.exists(fallback_dir):
            available_fallbacks = []
            for file in os.listdir(fallback_dir):
                if file.endswith((".png", ".jpg")):
                    fallback_path = os.path.join(fallback_dir, file)
                    if fallback_path not in used_fallbacks:
                        available_fallbacks.append(fallback_path)
            
            if available_fallbacks:
                selected_fallback = available_fallbacks[0]
                used_fallbacks.add(selected_fallback)
                print(f"🎨 Using project fallback image: {selected_fallback}")
                return selected_fallback
            
            # If all project fallbacks are used, cycle through them
            for file in os.listdir(fallback_dir):
                if file.endswith((".png", ".jpg")):
                    fallback_path = os.path.join(fallback_dir, file)
                    if os.path.exists(fallback_path):
                        print(f"🎨 Using project fallback image (cycling): {fallback_path}")
                        return fallback_path
        
        return None

    def create_video_with_ffmpeg(
        self,
        image_paths: List[str],
        audio_path: str,
        srt_path: str,
        output_path: str,
        audio_duration: float,
    ) -> str:
        """Create video using FFmpeg"""

        # Filter out None image paths and convert Windows paths to WSL paths
        valid_images = []
        used_fallbacks = set()  # Track used fallback images

        for img in image_paths:
            if img:
                # Convert Windows path to WSL path
                wsl_path = self.convert_windows_path_to_wsl(img)
                if os.path.exists(wsl_path):
                    # Ensure absolute path for FFmpeg
                    abs_path = os.path.abspath(wsl_path)
                    valid_images.append(abs_path)
                    print(f"✓ Found image: {abs_path}")
                else:
                    print(f"⚠ Original image not found: {wsl_path}")
                    # Try to find a unique fallback image in the same directory
                    fallback = self.find_fallback_image(img, used_fallbacks)
                    if fallback:
                        abs_fallback = os.path.abspath(fallback)
                        valid_images.append(abs_fallback)
                        print(f"✓ Using fallback instead: {abs_fallback}")
                    else:
                        print(f"✗ No fallback found for: {img}")

        if not valid_images:
            raise ValueError("No valid images found for video generation")

        # Create image list file for FFmpeg
        # Use the duration from audio generation step to ensure consistency
        duration_per_image = audio_duration / len(valid_images)
        print(f"📊 Using audio duration from generation step: {audio_duration:.2f}s")

        image_list_path = os.path.join(self.temp_dir, "images.txt")
        with open(image_list_path, "w") as f:
            for img_path in valid_images:
                f.write(f"file '{img_path}'\n")
                f.write(
                    f"duration {duration_per_image:.2f}\n"
                )  # Each image shows for calculated duration

        print(
            f"📊 Video timing: {len(valid_images)} images, {duration_per_image:.1f}s per image, total {audio_duration:.1f}s"
        )

        # Audio duration is already known, no need to recalculate
        
        # Default to vertical format (9:16) if not specified
        width, height = 1080, 1920  # Vertical format (mobile/social media)

        # Create main video from images
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
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2", # resize to proper aspect ratio
            temp_video_path,
        ]

        subprocess.run(ffmpeg_cmd, check=True)

        # Add subtitles
        video_with_subs = os.path.join(self.temp_dir, "video_with_subs.mp4")
        
        # Calculate dynamic subtitle styling based on video dimensions with smaller, centered text
        center_margin = 0  # Center vertically (restored)
        font_size = max(20, int(height * 0.012))  # Restored to previously working size: at least 20, or 1.2% of video height
        
        # Simplified subtitle styling for maximum compatibility and visibility
        subtitle_style = (
            f"FontSize={font_size},"
            f"PrimaryColour=&Hffffff,"  # White text
            f"OutlineColour=&H000000,"  # Black outline
            f"Outline=2,"               # Reduced outline thickness
            f"Alignment=5,"             # Middle center alignment (restored)
            f"MarginV={center_margin}," # Center margin (restored)
            f"Bold=1"                   # Bold text
        )

        subtitle_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_video_path,
            "-vf",
            f"subtitles={srt_path}:force_style='{subtitle_style}'",
            "-c:a",
            "copy",
            video_with_subs,
        ]

        subprocess.run(subtitle_cmd, check=True)

        # Add ending video (check local directory first)
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
            # First, re-encode ending video to match main video format (25fps)
            temp_ending = os.path.join(self.temp_dir, "ending_25fps.mp4")

            # Convert ending video to 25fps to match main video
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

            # Now concat with matching formats
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

    def create_video_with_image_script(
        self,
        image_paths: List[str],
        image_script: List[dict],
        audio_path: str,
        srt_path: str,
        output_path: str,
        audio_duration: float,
        aspect_ratio: str = "9:16",
    ) -> str:
        """Create video using image script with specific durations for each image"""
        # Set dimensions based on aspect ratio
        if aspect_ratio == "9:16":
            width, height = 1080, 1920  # Vertical format (mobile/social media)
        else:
            width, height = 1920, 1080  # Horizontal format (traditional)
        
        # Ensure we have the same number of images and script items
        if len(image_paths) != len(image_script):
            print(
                f"Warning: Mismatch between images ({len(image_paths)}) and script items ({len(image_script)})"
            )
            # Fallback to original method
            return self.create_video_with_ffmpeg(
                image_paths, audio_path, srt_path, output_path, audio_duration
            )

        # Filter and process image paths
        valid_images = []
        used_fallbacks = set()
        
        for img in image_paths:
            if img:
                # Convert Windows path to WSL path and ensure absolute path
                wsl_path = self.convert_windows_path_to_wsl(img)
                if os.path.exists(wsl_path):
                    abs_path = os.path.abspath(wsl_path)
                    valid_images.append(abs_path)
                    print(f"✓ Found image: {abs_path}")
                else:
                    print(f"⚠ Original image not found: {wsl_path}")
                    fallback = self.find_fallback_image(img, used_fallbacks)
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
        
        # Calculate dynamic subtitle styling based on video dimensions with smaller, centered text
        center_margin = 0  # Center vertically (restored)
        font_size = max(20, int(height * 0.012))  # Restored to previously working size: at least 20, or 1.2% of video height
        
        # Simplified subtitle styling for maximum compatibility and visibility
        subtitle_style = (
            f"FontSize={font_size},"
            f"PrimaryColour=&Hffffff,"  # White text
            f"OutlineColour=&H000000,"  # Black outline
            f"Outline=2,"               # Reduced outline thickness
            f"Alignment=5,"             # Middle center alignment (restored)
            f"MarginV={center_margin}," # Center margin (restored)
            f"Bold=1"                   # Bold text
        )

        subtitle_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_video_path,
            "-vf",
            f"subtitles={srt_path}:force_style='{subtitle_style}'",
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
        overlay_text: str = "Demo Video",
        blur_strength: int = 24,
        aspect_ratio: str = "9:16",  # "9:16" for vertical, "16:9" for horizontal
    ) -> str:
        """Create video with blur background effect, custom aspect ratio, and text overlay
        
        Args:
            image_paths: List of image file paths
            image_script: List of dictionaries with image descriptions and durations
            audio_path: Path to audio file
            srt_path: Path to SRT subtitle file
            output_path: Path for output video
            audio_duration: Duration of audio in seconds
            overlay_text: Text to overlay on video
            blur_strength: Gaussian blur strength (default: 24)
            aspect_ratio: Video aspect ratio - "9:16" for vertical, "16:9" for horizontal
        """
        # Set dimensions based on aspect ratio
        if aspect_ratio == "9:16":
            width, height = 1080, 1920  # Vertical format (mobile/social media)
        else:
            width, height = 1920, 1080  # Horizontal format (traditional)
        
        print(f"🎬 Creating video with blur background effect")
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
                wsl_path = self.convert_windows_path_to_wsl(img)
                if os.path.exists(wsl_path):
                    abs_path = os.path.abspath(wsl_path)
                    valid_images.append(abs_path)
                    print(f"✓ Found image: {abs_path}")
                else:
                    print(f"⚠ Original image not found: {wsl_path}")
                    fallback = self.find_fallback_image(img, used_fallbacks)
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

        print(f"📊 Video timing: {len(valid_script_pairs)} images with blur background effect")
        print(f"📊 Image total duration: {total_image_duration:.2f}s")
        print(f"📊 Audio duration: {audio_duration:.2f}s")

        # Create main video with blur background effect
        temp_video_path = os.path.join(self.temp_dir, "temp_video_blur.mp4")
        
        # Complex filter that mimics the batch file:
        # 1. Create blurred background: scale to fill frame, crop, blur
        # 2. Scale original to fit within frame while maintaining aspect ratio  
        # 3. Center the original image on the blurred background
        filter_complex = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},gblur={blur_strength}[blurred];"
            f"[0:v]scale={min(width, height)}:{min(width, height)}:force_original_aspect_ratio=decrease[scaled];"
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

        print("🎬 Executing FFmpeg with blur background effect...")
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
        # Enhanced subtitle styling positioned at the center of the screen with smaller text
        # Calculate center margin to position subtitles in the middle
        center_margin = 0  # Center vertically (restored)
        # Use much smaller font size for subtle subtitles (90% reduction from original)
        font_size = max(18, int(height * 0.01))  # Restored to previously working size: at least 18, or 1% of video height
        
        # Simplified subtitle styling for maximum compatibility and visibility
        subtitle_style = (
            f"FontSize={font_size},"
            f"PrimaryColour=&Hffffff,"  # White text
            f"OutlineColour=&H000000,"  # Black outline
            f"Outline=2,"               # Reduced outline thickness
            f"Alignment=5,"             # Middle center alignment (restored)
            f"MarginV={center_margin}," # Center margin (restored)
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

        print(f"✅ Successfully created blur background video: {output_path}")
        print(f"📐 Format: {aspect_ratio} ({width}x{height})")
        print(f"🌫️  Blur effect applied with strength {blur_strength}")
        
        return output_path

    def create_video_from_prepared_data(
        self,
        transcript: str,
        audio_path: str,
        output_path: str = "output_video.mp4",
        custom_srt_path: str = None,
        use_blur_background: bool = False,
        overlay_text: str = "Demo Video",
        blur_strength: int = 24,
        aspect_ratio: str = "9:16",
    ) -> str:
        """
        Create video from already-prepared data (transcript, audio, images)
        
        This method does NOT regenerate any content - it uses what was already prepared
        in the optimization step to create the final video efficiently.
        
        Args:
            transcript: Already generated transcript text
            audio_path: Path to already generated audio file
            output_path: Path for the output video
            custom_srt_path: Path to custom SRT file (optional)
            use_blur_background: Whether to use blur background effect
            overlay_text: Text to overlay on video (for blur effect)
            blur_strength: Blur strength (integer, default 24)
            aspect_ratio: Video aspect ratio ("9:16" or "16:9")
            
        Returns:
            Path to generated video file
        """
        print("🚀 Creating video from prepared data (streamlined process)")
        print(f"📝 Transcript length: {len(transcript)} characters")
        print(f"🎵 Audio file: {audio_path}")
        print(f"📐 Aspect ratio: {aspect_ratio}")
        
        # Set dimensions based on aspect ratio
        if aspect_ratio == "16:9":
            width, height = 1920, 1080
        else:  # 9:16 (vertical)
            width, height = 1080, 1920
            
        print(f"📐 Video dimensions: {width}x{height}")

        # Step 1: Get audio duration
        import librosa
        audio_duration = librosa.get_duration(path=audio_path)
        print(f"🎵 Audio duration: {audio_duration:.2f} seconds")

        # Step 2: Generate SRT file from transcript (only if no custom SRT provided)
        if custom_srt_path:
            srt_path = custom_srt_path
            print(f"📄 Using custom SRT: {srt_path}")
        else:
            print("Step 2: Generating SRT file from prepared transcript...")
            # Convert transcript to sentences for SRT generation
            import re
            sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
            srt_path = self.generate_srt_file(sentences, audio_duration)
            print(f"📄 Generated SRT: {srt_path}")

        # Step 3: Use existing image search (this will reuse the same logic but with prepared transcript)
        print("Step 3: Searching for images using prepared transcript...")
        
        # Generate image script for timing
        try:
            image_script = self.get_image_list_script(transcript, audio_duration)
            if image_script:
                descriptions = [item["description"] for item in image_script]
                print(f"✓ Using {len(descriptions)} generated image descriptions")
            else:
                # Fallback to sentences
                import re
                descriptions = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
                print(f"✓ Using {len(descriptions)} sentences as descriptions")
        except Exception as e:
            print(f"⚠ Image script generation failed, using sentences: {e}")
            import re
            descriptions = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]

        # Search for images (this is the only unavoidable search since images depend on the final transcript)
        image_paths = self.vector_search_images(descriptions)
        
        # Step 4: Create video
        print("Step 4: Creating final video...")
        
        if use_blur_background and image_script and len(image_paths) == len(image_script):
            print("🌫️  Creating video with blur background effect...")
            video_path = self.create_video_with_blur_background(
                image_paths, image_script, audio_path, srt_path, output_path, 
                audio_duration, overlay_text, blur_strength, aspect_ratio
            )
        elif image_script and len(image_paths) == len(image_script):
            print("🎬 Creating video with image script timing...")
            video_path = self.create_video_with_image_script(
                image_paths, image_script, audio_path, srt_path, output_path, audio_duration
            )
        else:
            print("🎬 Creating video with standard timing...")
            video_path = self.create_video_with_ffmpeg(
                image_paths, audio_path, srt_path, output_path, audio_duration
            )
        
        print(f"✅ Video created successfully: {video_path}")
        return video_path

    def create_video_from_prepared_data_with_images(
        self,
        transcript: str,
        audio_path: str,
        output_path: str = "output_video.mp4",
        prepared_image_paths: List[str] = None,
        prepared_descriptions: List[str] = None,
        custom_srt_path: str = None,
        use_blur_background: bool = False,
        overlay_text: str = "Demo Video",
        blur_strength: int = 24,
        aspect_ratio: str = "9:16",
    ) -> str:
        """
        Create video from already-prepared data including prepared image assignments
        
        This method uses the exact same images that were shown in the preview,
        ensuring consistency between preview and final video.
        
        Args:
            transcript: Already generated transcript text
            audio_path: Path to already generated audio file
            output_path: Path for the output video
            prepared_image_paths: List of image paths from the preparation step
            prepared_descriptions: List of descriptions from the preparation step
            custom_srt_path: Path to custom SRT file (optional)
            use_blur_background: Whether to use blur background effect
            overlay_text: Text to overlay on video (for blur effect)
            blur_strength: Blur strength (integer, default 24)
            aspect_ratio: Video aspect ratio ("9:16" or "16:9")
            
        Returns:
            Path to generated video file
        """
        print("🚀 Creating video from prepared data with exact image assignments")
        print(f"📝 Transcript length: {len(transcript)} characters")
        print(f"🎵 Audio file: {audio_path}")
        print(f"🖼️  Using {len(prepared_image_paths)} prepared image assignments")
        print(f"📐 Aspect ratio: {aspect_ratio}")
        
        # Set dimensions based on aspect ratio
        if aspect_ratio == "16:9":
            width, height = 1920, 1080
        else:  # 9:16 (vertical)
            width, height = 1080, 1920
            
        print(f"📐 Video dimensions: {width}x{height}")

        # Step 1: Get audio duration
        import librosa
        audio_duration = librosa.get_duration(path=audio_path)
        print(f"🎵 Audio duration: {audio_duration:.2f} seconds")

        # Step 2: Generate SRT file from transcript (only if no custom SRT provided)
        if custom_srt_path:
            srt_path = custom_srt_path
            print(f"📄 Using custom SRT: {srt_path}")
        else:
            print("Step 2: Generating SRT file from prepared transcript...")
            # Convert transcript to sentences for SRT generation
            import re
            sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
            srt_path = self.generate_srt_file(sentences, audio_duration)
            print(f"📄 Generated SRT: {srt_path}")

        # Step 3: Use prepared image assignments (NO SEARCH!)
        print("Step 3: Using prepared image assignments (no new search)...")
        image_paths = prepared_image_paths
        descriptions = prepared_descriptions
        
        print(f"✓ Using {len(image_paths)} prepared images")
        print(f"✓ Using {len(descriptions)} prepared descriptions")
        
        # Step 4: Create video with prepared images
        print("Step 4: Creating final video with prepared images...")
        
        # Generate image script for timing (reuse prepared descriptions)
        try:
            image_script = self.get_image_list_script(transcript, audio_duration)
            if image_script and len(image_script) == len(image_paths):
                print(f"✓ Using {len(image_script)} timed image segments")
            else:
                # Fallback: create basic timing from prepared descriptions
                segment_duration = audio_duration / max(len(descriptions), 1)
                image_script = []
                for i, desc in enumerate(descriptions):
                    image_script.append({
                        "description": desc,
                        "duration": segment_duration
                    })
                print(f"✓ Using fallback timing: {segment_duration:.2f}s per segment")
        except Exception as e:
            print(f"⚠ Image script generation failed, using equal timing: {e}")
            segment_duration = audio_duration / max(len(descriptions), 1)
            image_script = []
            for i, desc in enumerate(descriptions):
                image_script.append({
                    "description": desc,
                    "duration": segment_duration
                })
        
        if use_blur_background and image_script and len(image_paths) == len(image_script):
            print("🌫️  Creating video with blur background effect...")
            video_path = self.create_video_with_blur_background(
                image_paths, image_script, audio_path, srt_path, output_path, 
                audio_duration, overlay_text, blur_strength, aspect_ratio
            )
        elif image_script and len(image_paths) == len(image_script):
            print("🎬 Creating video with image script timing...")
            video_path = self.create_video_with_image_script(
                image_paths, image_script, audio_path, srt_path, output_path, audio_duration
            )
        else:
            print("🎬 Creating video with standard timing...")
            video_path = self.create_video_with_ffmpeg(
                image_paths, audio_path, srt_path, output_path, audio_duration
            )
        
        print(f"✅ Video created successfully with prepared images: {video_path}")
        return video_path

    def generate_video_from_article(
        self,
        article: str = None,
        web_link: str = None,
        output_path: str = "output_video.mp4",
        use_gpt_transcript: bool = False,
        custom_srt_path: str = None,
        custom_audio_path: str = None,
        use_blur_background: bool = False,
        overlay_text: str = "Demo Video",
        blur_strength: int = 24,
        aspect_ratio: str = "9:16",
    ) -> str:
        """Main function to generate video from article

        Args:
            article: The article content (optional if custom_srt_path is provided)
            output_path: Path for the output video
            use_gpt_transcript: Whether to generate a transcript using GPT first
            custom_srt_path: Path to custom SRT file (optional)
            custom_audio_path: Path to custom audio file (optional)
            use_blur_background: Whether to use blur background effect
            overlay_text: Text to overlay on video (when using blur background)
            blur_strength: Gaussian blur strength for background (default: 24)
            aspect_ratio: Video aspect ratio - "9:16" for vertical, "16:9" for horizontal
        """
        try:
            # Handle SRT generation or use custom SRT
            if custom_srt_path and os.path.exists(custom_srt_path):
                print("Using custom SRT file provided by user...")
                srt_path = custom_srt_path
                # Read sentences from SRT for image search
                sentences = self._extract_sentences_from_srt(custom_srt_path)
            else:
                # Need article or web_link for default flow
                if web_link:
                    print("Step 0a: Extracting content from web link...")
                    article = self.extract_web_content(web_link)
                    print(f"Extracted article length: {len(article)} characters")
                elif not article:
                    raise ValueError(
                        "Either article content or web_link is required when not using custom SRT"
                    )

                # Optional: Generate transcript from article using GPT
                if use_gpt_transcript:
                    print("Step 0b: Generating transcript from article using GPT...")
                    transcript = self.generate_transcript_from_article(article)
                    print("Transcript generated successfully")
                else:
                    transcript = article

                print("Step 1: Splitting article into sentences...")
                sentences = self.split_article_into_sentences(transcript)
                print(f"Found {len(sentences)} sentences")
                print(article)
                print(transcript)
                print(sentences)

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

            # Generate image list script with durations
            print("Step 3: Generating image list script with durations...")
            image_script = self.get_image_list_script(transcript, audio_duration)
            if image_script:
                print(f"Generated script with {len(image_script)} image descriptions")
                for i, item in enumerate(image_script):
                    print(f"  {i+1}. {item['description']} - {item['duration']:.1f}s")
            else:
                print("Warning: Failed to generate image script")

            # Generate SRT if not using custom
            if not custom_srt_path:
                print("Step 4: Generating SRT file...")
                srt_path = self.generate_srt_file(sentences, audio_duration)
                print(f"SRT file created: {srt_path}")

            print("Step 5: Searching for images using vector search...")
            if image_script:
                # Use image descriptions from the generated script
                descriptions = [item["description"] for item in image_script]
                image_paths = self.vector_search_images(descriptions)
                print(
                    f"Found {len([p for p in image_paths if p])} valid images from {len(descriptions)} descriptions"
                )
            else:
                # Fallback to sentences if image script generation failed
                print("Warning: Using sentences as fallback for image search")
                image_paths = self.vector_search_images(sentences)
                print(
                    f"Found {len([p for p in image_paths if p])} valid images from sentences"
                )

            print("Step 6: Creating video with FFmpeg...")
            # Choose video creation method based on options
            if use_blur_background:
                print("🌫️  Using blur background effect for video creation...")
                if image_script and len(image_paths) == len(image_script):
                    final_video = self.create_video_with_blur_background(
                        image_paths,
                        image_script,
                        audio_path,
                        srt_path,
                        output_path,
                        audio_duration,
                        overlay_text,
                        blur_strength,
                        aspect_ratio,
                    )
                else:
                    # Create basic image script for blur background
                    print("Creating basic image script for blur background...")
                    duration_per_image = audio_duration / len(image_paths)
                    basic_script = [{"description": f"image_{i}", "duration": duration_per_image} 
                                  for i in range(len(image_paths))]
                    final_video = self.create_video_with_blur_background(
                        image_paths,
                        basic_script,
                        audio_path,
                        srt_path,
                        output_path,
                        audio_duration,
                        overlay_text,
                        blur_strength,
                        aspect_ratio,
                    )
            elif image_script and len(image_paths) == len(image_script):
                # Use the script with durations for video creation
                final_video = self.create_video_with_image_script(
                    image_paths,
                    image_script,
                    audio_path,
                    srt_path,
                    output_path,
                    audio_duration,
                    aspect_ratio,
                )
            else:
                # Fallback to original method if script doesn't match
                final_video = self.create_video_with_ffmpeg(
                    image_paths, audio_path, srt_path, output_path, audio_duration
                )

            # Collect image preview information
            image_previews = self._collect_image_previews(image_paths, image_script if image_script else sentences)
            
            print(f"Video generation completed: {final_video}")
            return {
                "video_path": final_video,
                "image_previews": image_previews,
                "total_images": len([p for p in image_paths if p])
            }

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
   The stars have always guided us… but what if they could speak directly to you?
Meet Astro — your personal AI astrologer, combining ancient wisdom with cutting-edge technology.

Astro analyzes your unique birth chart in seconds, offering insights on love, career, and personal growth tailored just for you. No generic horoscopes here — every prediction, every piece of guidance is based on your personal cosmic blueprint. With 24/7 access, Astro learns your patterns, answers your questions, and helps you navigate life’s big decisions in real time.

Whether you’re seeking clarity, confidence, or just a little magic, Astro is here to illuminate your path. The universe is talking. Are you ready to listen? Visit Astro.ai and start your journey today.    """

    # Initialize video generator (will use environment variables for Pinecone)
    generator = VideoGenerator()

    try:
        # custom_audio = "data/media/somer_smaple.wav"
        # Example 1: Generate audio from text using API
        print("=== Example 1: Generate audio from text ===")
        output_file = generator.generate_video_from_article(
            # web_link="https://www.taaze.tw/products/11101058474.html",
            article=fixed_article,
            use_gpt_transcript=True,
            output_path="New_pre-market.mp4",
            custom_srt_path=None,
            custom_audio_path=None,  # Will use API to generate audio
        )
        print(f"Success! Video with generated audio saved to: {output_file}")

        # print("=== Example 2: Generate audio from text ===")
        # output_file = generator.generate_video_from_article(
        #     web_link="https://www.taaze.tw/products/11101058474.html",
        #     # article=fixed_article,
        #     use_gpt_transcript=True,
        #     output_path="book_video.mp4",
        #     custom_srt_path=None,
        #     custom_audio_path=None,  # Will use API to generate audio
        # )
        # print(f"Success! Video with generated audio saved to: {output_file}")


        # Example 3: Use custom audio file (if available)
        # custom_audio = "data/media/raymond_bridge8-3.mp3"
        # if os.path.exists(custom_audio):
        #     print("\n=== Example 3: Use custom audio file ===")
        #     output_file_2 = generator.generate_video_from_article(
        #         article=fixed_article,
        #         output_path="raymond_bridge11.mp4",
        #         use_gpt_transcript=False,
        #         custom_srt_path=None,
        #         custom_audio_path=custom_audio,
        #     )
        #     print(f"Success! Video with custom audio saved to: {output_file_2}")

    except Exception as e:
        print(f"Error: {e}")
    # finally:
    #     generator.cleanup()


if __name__ == "__main__":
    main()
