# Realtime Video Generator Documentation

## Overview

The `Realtime_Video_Gen.py` script provides a comprehensive pipeline for generating videos from text content using ComfyUI integration and AI-powered image generation. It combines Azure OpenAI for content processing, ComfyUI for image/video generation, and FFmpeg for video compilation.

## Class: RealtimeVideoGenerator

### Initialization

```python
generator = RealtimeVideoGenerator(comfyui_url="https://your-comfyui-url/api/prompt")
```

**Parameters:**
- `comfyui_url` (str, optional): ComfyUI API endpoint URL. Defaults to environment variable `COMFYUI_URL`

### Core Features

#### 1. AI-Powered Art Style Selection

The system uses a dual-prompt approach to generate contextually appropriate visuals:

**First Stage: Art Style Selection**
- Analyzes content to determine appropriate visual style
- Specialized categories:
  - **Children's Content**: Watercolor, cartoon, chibi styles with vivid colors
  - **Educational Content**: Flat vector, minimalist, technical drawing styles  
  - **Novels**: Painterly, oil painting, fantasy concept art styles

**Second Stage: Image Prompt Generation**
- Creates detailed image prompts incorporating the selected art style
- Optimized for AI image generation tools

#### 2. Text-to-Video Pipeline

```python
video_path = generator.process_article_to_video(
    article="Your article text...",
    output_path="output_video.mp4",
    target_duration=60,
    use_comfyui=True,
    aspect_ratio="9:16"
)
```

**Parameters:**
- `article` (str): Input text content
- `output_path` (str): Output video file path
- `target_duration` (int): Target duration in seconds
- `use_comfyui` (bool): Whether to use ComfyUI for image generation
- `aspect_ratio` (str): Video aspect ratio ("9:16" or "16:9")

### Key Methods

#### Content Processing

```python
# Generate art style and image prompt from text
result = generator.process_text_to_image_prompt(text)
# Returns: {'transcript': text, 'art_style': style, 'image_prompt': prompt}

# Break script into timed scenes
scenes = generator.break_script_into_scenes(script, duration)
# Returns: [{'description': 'scene description', 'duration': 3.5}, ...]
```

#### ComfyUI Integration

```python
# Generate single comprehensive image
image_path = generator.generate_single_image_with_comfyui(transcript, art_style)

# Generate complete video (experimental)
video_path = generator.generate_video_with_comfyui(transcript, art_style, duration)
```

#### Completion Detection System

The system includes a robust "heartbeat" detection mechanism for monitoring ComfyUI generation:

```python
# Heartbeat-based completion detection
is_complete = generator.check_comfyui_completion_heartbeat(
    prompt_id="comfyui_prompt_id", 
    output_folder="output_folder_name",
    max_attempts=30
)
```

**Features:**
- **Dual Verification**: Checks both API status and file system
- **File Age Validation**: Only considers recently created files (within 10 minutes)
- **Fallback Logic**: Trusts file system over API after 70% of attempts
- **Multiple Path Checking**: Searches common ComfyUI output locations

#### Audio Transcription (Placeholder)

```python
# Currently returns placeholder text - implement with Whisper API
transcript = generator.transcribe_audio("audio_file.mp3")
```

### Workflow Examples

#### Example 1: Full Pipeline with ComfyUI

```python
generator = RealtimeVideoGenerator()

# Complete pipeline: text → art style → ComfyUI image → video
video_path = generator.process_article_to_video(
    article="AI is transforming our world...",
    output_path="ai_video.mp4",
    target_duration=30,
    use_comfyui=True,
    aspect_ratio="9:16"
)
```

#### Example 2: Fast Generation with Database Images

```python
# Skip ComfyUI generation, use existing images
video_path = generator.process_article_to_video(
    article="Your content...",
    output_path="fast_video.mp4",
    target_duration=30,
    use_comfyui=False,  # Use database images instead
    aspect_ratio="16:9"
)
```

#### Example 3: Individual Component Testing

```python
# Test art style selection
result = generator.process_text_to_image_prompt("Your text")
print(f"Selected style: {result['art_style']}")

# Test scene breakdown
scenes = generator.break_script_into_scenes("Script text", 60)
print(f"Created {len(scenes)} scenes")
```

### Configuration Requirements

#### Environment Variables

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT="your-endpoint"
AZURE_OPENAI_API_KEY="your-key"

# ComfyUI
COMFYUI_URL="https://your-comfyui-url/api/prompt"

# ElevenLabs (for audio generation)
ELEVENLABS_API_KEY="your-key"
```

#### Dependencies

```python
# Core dependencies
from langchain_openai import AzureChatOpenAI
import librosa
import requests
import tempfile

# Project modules
from .image_generator import generate_image_prompt_fun, call_image_request_function
from .video_generator import VideoGenerator
```

### Output Management

#### File Paths
The system manages multiple output paths for ComfyUI integration:

```python
output_base_paths = [
    "/mnt/c/Users/x7048/Documents/ComfyUI/output/{output_folder}",
    "/mnt/c/Users/x7048/Documents/ComfyUI/output",
    "/mnt/c/Users/x7048/Documents/ComfyUI/output/generated/",
    # ... additional paths
]
```

#### Generated Files
- **Images**: PNG, JPG, JPEG formats
- **Videos**: MP4, MOV, AVI formats
- **Temporary Files**: Automatically cleaned up via `cleanup()` method

### Error Handling and Fallbacks

The system includes comprehensive fallback mechanisms:

1. **ComfyUI Failure**: Falls back to database image search
2. **Art Style Selection Failure**: Uses "Photorealistic" default
3. **Scene Breakdown Failure**: Creates basic time-based scenes
4. **File Detection Issues**: Multiple path checking with time validation

### Performance Considerations

- **Single Image Strategy**: Uses one comprehensive image for all scenes when ComfyUI is enabled
- **Heartbeat Optimization**: 3-second intervals between completion checks
- **Timeout Management**: Scales wait time with video duration
- **Resource Cleanup**: Automatic temporary file management

### Integration with API Server

The class is designed to work with the Flask API server (`api_server.py`) through the `RealtimeVideoGeneratorAdapter`:

```python
# Used in API endpoints for real-time video generation
generator = RealtimeVideoGeneratorAdapter(comfyui_url=comfyui_url)
```

### Future Enhancements

1. **Audio Transcription**: Replace placeholder with Whisper API integration
2. **Video Generation**: Enhance ComfyUI video workflows
3. **Style Customization**: Allow custom art style prompts
4. **Batch Processing**: Support multiple articles simultaneously

### Usage Notes

- Ensure ComfyUI server is running and accessible
- Monitor ComfyUI output directories for proper file permissions
- Consider video duration when setting timeout values
- Use vertical aspect ratio (9:16) for social media content
- Test individual components before running full pipeline

### Cleanup

Always call cleanup to remove temporary files:

```python
try:
    video_path = generator.process_article_to_video(...)
finally:
    generator.cleanup()
```