#!/usr/bin/env python3
"""
SRT Manager
Handles subtitle generation, text width calculation, and line breaking for videos
"""

import os
import re
import tempfile
from typing import List


class SRTManager:
    """Manager class for handling SRT subtitle generation and text processing"""
    
    def __init__(self, temp_dir: str = None):
        """Initialize the SRT manager
        
        Args:
            temp_dir: Temporary directory for file operations
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp()
    
    def _calculate_text_width(self, text: str) -> int:
        """Calculate display width of text considering Chinese and English characters"""
        width = 0
        for char in text:
            if "\u4e00" <= char <= "\u9fff":  # Chinese characters
                width += 2
            else:  # English and other characters
                width += 1
        return width

    def _break_long_lines(self, text: str, max_width: int = 36, aspect_ratio: str = "vertical") -> List[str]:
        """Break long text into multiple lines based on character width with smart chunking"""
        # Adjust max_width based on aspect ratio
        if aspect_ratio == "horizontal":
            max_width = 50  # More characters allowed for horizontal videos
        else:
            max_width = 25  # Fewer characters for vertical videos
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

    def generate_srt_file(self, sentences: List[str], audio_duration: float, aspect_ratio: str = "vertical") -> str:
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
            lines = self._break_long_lines(sentence, aspect_ratio=aspect_ratio)

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

            calculated_duration = max(
                min_duration, sentence_chars / chars_per_second * 0.9
            )  # Slightly faster timing

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

    def extract_sentences_from_srt(self, srt_path: str) -> List[str]:
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