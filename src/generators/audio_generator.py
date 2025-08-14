#!/usr/bin/env python3
"""
Audio Generator Module
Handles audio generation from scripts using ElevenLabs API
"""

import os
import tempfile
import librosa
from typing import Tuple, Optional
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AudioGenerator:
    def __init__(self):
        """Initialize the audio generator with ElevenLabs API"""
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        
        if not self.api_key:
            raise ValueError("ElevenLabs API key not found in environment variables")
        
        self.client = ElevenLabs(api_key=self.api_key)
        self.voice_id = "JBFqnCBsd6RMkjVDRZzb"  # Default voice
        self.output_format = "mp3_44100_128"
    
    def generate_audio(
        self, 
        script: str, 
        voice_id: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Generate audio from script text
        
        Args:
            script: Text script to convert to audio
            voice_id: ElevenLabs voice ID (optional, uses default if not provided)
            output_path: Custom output path (optional, creates temp file if not provided)
            
        Returns:
            Tuple of (audio_file_path, duration_in_seconds)
        """
        try:
            # Use provided voice or default
            voice_to_use = voice_id or self.voice_id
            
            print(f"Generating audio with ElevenLabs API (text length: {len(script)} chars)...")
            
            # Generate audio using ElevenLabs
            audio_generator = self.client.generate(
                text=script,
                voice=voice_to_use,
                output_format=self.output_format,
                model="eleven_multilingual_v2"
            )
            
            # Create output file path
            if not output_path:
                temp_fd, output_path = tempfile.mkstemp(suffix=".mp3")
                os.close(temp_fd)
            
            # Write audio data to file
            with open(output_path, "wb") as audio_file:
                for chunk in audio_generator:
                    audio_file.write(chunk)
            
            # Get audio duration
            duration = librosa.get_duration(path=output_path)
            
            print(f"✓ Generated audio from ElevenLabs API, duration: {duration:.2f}s")
            
            return output_path, duration
            
        except Exception as e:
            raise Exception(f"Failed to generate audio: {str(e)}")
    
    def get_available_voices(self) -> list:
        """
        Get list of available voices from ElevenLabs
        
        Returns:
            List of voice information dictionaries
        """
        try:
            voices = self.client.voices.get_all()
            
            voice_list = []
            for voice in voices.voices:
                voice_info = {
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "category": voice.category,
                    "description": getattr(voice, 'description', ''),
                    "preview_url": getattr(voice, 'preview_url', None)
                }
                voice_list.append(voice_info)
            
            return voice_list
            
        except Exception as e:
            raise Exception(f"Failed to get available voices: {str(e)}")
    
    def validate_audio_settings(self, script: str, voice_id: str = None) -> dict:
        """
        Validate audio generation settings
        
        Args:
            script: Script text
            voice_id: Voice ID to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            "valid": True,
            "warnings": [],
            "estimates": {}
        }
        
        # Validate script
        if not script or not script.strip():
            result["valid"] = False
            result["warnings"].append("Script is empty")
            return result
        
        # Estimate audio characteristics
        word_count = len(script.split())
        char_count = len(script)
        estimated_duration = word_count / 150 * 60  # 150 words per minute average
        
        result["estimates"] = {
            "duration_seconds": round(estimated_duration, 2),
            "file_size_mb": round(estimated_duration * 0.5, 2),  # Rough estimate for MP3
            "character_count": char_count
        }
        
        # Check for potential issues
        if char_count > 5000:
            result["warnings"].append("Script is very long, may take time to generate")
        
        if estimated_duration < 5:
            result["warnings"].append("Very short audio, may not be suitable for video")
        elif estimated_duration > 180:
            result["warnings"].append("Very long audio, consider breaking into segments")
        
        # Validate voice ID
        if voice_id and voice_id != self.voice_id:
            try:
                voices = self.get_available_voices()
                voice_ids = [v["voice_id"] for v in voices]
                if voice_id not in voice_ids:
                    result["warnings"].append(f"Voice ID '{voice_id}' not found, will use default")
            except:
                result["warnings"].append("Could not validate voice ID")
        
        return result
    
    def get_audio_info(self, audio_path: str) -> dict:
        """
        Get information about an existing audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio file information
        """
        try:
            if not os.path.exists(audio_path):
                raise Exception("Audio file not found")
            
            # Get audio duration and other properties
            duration = librosa.get_duration(path=audio_path)
            file_size = os.path.getsize(audio_path)
            
            # Load audio to get more details
            y, sr = librosa.load(audio_path, sr=None)
            
            info = {
                "duration_seconds": round(duration, 2),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "sample_rate": sr,
                "channels": 1 if len(y.shape) == 1 else y.shape[0],
                "samples": len(y) if len(y.shape) == 1 else y.shape[1]
            }
            
            return info
            
        except Exception as e:
            raise Exception(f"Failed to get audio info: {str(e)}")