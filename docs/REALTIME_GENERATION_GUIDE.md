# Real-Time Video Generation Guide

## Overview

The enhanced `RealtimeVideoGenerator` now supports real-time image generation with a dual-prompt system:

- **Prompt 1**: Art Style Selection (runs once per session)
- **Prompt 2**: Image Prompt Generation (runs for each image needed)
- **Timestamps**: Automatic video duration timing
- **Storage**: All prompts stored for ComfyUI batch processing

## Key Features

### 🎨 Dual Prompt System
1. **Art Style Selection**: Analyzes user input once to select appropriate visual style
2. **Image Generation**: Creates detailed prompts for each required image
3. **Smart Caching**: Art style is reused until manually reset

### ⏱️ Timestamp Management
- Automatic calculation based on video duration
- Configurable images per second (default: 0.5 = 1 image every 2 seconds)
- Precise timing for video synchronization

### 📦 Batch Processing
- All prompts stored in memory for ComfyUI
- Efficient batch submission and monitoring
- Heartbeat detection for completion tracking

## Quick Start

```python
from src.generators.Realtime_Video_Gen import RealtimeVideoGenerator

# Initialize
generator = RealtimeVideoGenerator()

# Process user input
result = generator.process_user_input_realtime(
    user_input="A beautiful sunset over mountains",
    video_duration=30  # 30 second video
)

# Get prompts for ComfyUI
prompts = generator.get_stored_prompts_for_comfyui()
timestamps = generator.get_image_timestamps()

# Generate images with ComfyUI
image_paths = generator.batch_generate_images_with_comfyui(result['image_data'])
```

## API Reference

### Main Methods

#### `process_user_input_realtime(user_input, video_duration)`
Main entry point for real-time generation.

**Parameters:**
- `user_input` (str): User's text input
- `video_duration` (int): Target video duration in seconds

**Returns:**
```python
{
    "user_input": str,
    "art_style": str,
    "video_duration": int,
    "total_images": int,
    "image_data": List[Dict],
    "comfyui_prompts": List[str],
    "timestamps": List[float]
}
```

#### `generate_realtime_images(user_input, duration, images_per_second)`
Generate image prompts with timestamps.

**Parameters:**
- `user_input` (str): User's text input
- `video_duration` (int): Video duration in seconds
- `images_per_second` (float): Images per second (default: 0.5)

**Returns:**
List of image data dictionaries with prompts and timestamps.

#### `batch_generate_images_with_comfyui(image_data)`
Send all prompts to ComfyUI for batch generation.

**Parameters:**
- `image_data` (List[Dict]): Image data from `generate_realtime_images`

**Returns:**
List of generated image file paths.

### Utility Methods

#### `get_stored_prompts_for_comfyui()`
Returns: `List[str]` - All stored image prompts

#### `get_image_timestamps()`
Returns: `List[float]` - Timestamps in seconds

#### `reset_realtime_state()`
Reset art style and prompts for new session.

## Usage Examples

### Example 1: Basic Real-Time Generation

```python
generator = RealtimeVideoGenerator()

# Process user input
result = generator.process_user_input_realtime(
    "A cat playing with yarn in a cozy room",
    video_duration=20
)

print(f"Art Style: {result['art_style']}")
print(f"Generated {result['total_images']} images")
print(f"Timestamps: {result['timestamps']}")
```

### Example 2: Multiple Sessions with Style Reuse

```python
generator = RealtimeVideoGenerator()

# First input - selects art style
result1 = generator.process_user_input_realtime(
    "A magical forest with fairy lights",
    video_duration=15
)

# Second input - reuses same art style
result2 = generator.process_user_input_realtime(
    "Unicorns dancing in the moonlight", 
    video_duration=15
)

# Both will use the same art style since content is similar
```

### Example 3: Reset for Different Content Types

```python
generator = RealtimeVideoGenerator()

# Children's content
result1 = generator.process_user_input_realtime(
    "A friendly dragon reading stories",
    video_duration=20
)

# Reset for educational content
generator.reset_realtime_state()

# Educational content - will select different art style
result2 = generator.process_user_input_realtime(
    "The water cycle and precipitation",
    video_duration=25
)
```

### Example 4: Full Pipeline with ComfyUI

```python
generator = RealtimeVideoGenerator()

# Generate prompts
result = generator.process_user_input_realtime(
    "Space exploration and distant galaxies",
    video_duration=30
)

# Generate actual images
image_paths = generator.batch_generate_images_with_comfyui(result['image_data'])

# Use images for video compilation
# (integrate with existing video generation pipeline)
```

## Configuration Options

### Images Per Second
Control how many images are generated:

```python
# More images (every 1 second)
result = generator.generate_realtime_images(
    user_input="...",
    video_duration=30,
    images_per_second=1.0
)

# Fewer images (every 3 seconds)  
result = generator.generate_realtime_images(
    user_input="...",
    video_duration=30,
    images_per_second=0.33
)
```

### Art Style Categories
The system automatically selects from different style categories:

- **Children's Content**: Watercolor, Cartoon, Anime, etc.
- **Educational Content**: Flat Vector, Minimalist, Technical, etc.
- **Novel/Story Content**: Oil Painting, Fantasy Art, Impressionist, etc.

## Integration with Existing Pipeline

The real-time system integrates seamlessly with existing video generation:

```python
# Real-time generation
generator = RealtimeVideoGenerator()
result = generator.process_user_input_realtime(user_input, duration=30)

# Generate images
image_paths = generator.batch_generate_images_with_comfyui(result['image_data'])

# Use existing video compilation
video_path = generator.compile_video(
    image_paths=image_paths,
    scenes=result['image_data'],  # Contains duration info
    script=user_input,
    output_path="realtime_video.mp4"
)
```

## Performance Tips

1. **Reuse Art Styles**: Don't reset state unnecessarily to avoid re-selecting styles
2. **Batch Processing**: Use `batch_generate_images_with_comfyui()` for efficient generation
3. **Adjust Image Count**: Lower `images_per_second` for faster processing
4. **Monitor Resources**: ComfyUI generation is resource-intensive

## Troubleshooting

### Common Issues

1. **No Art Style Selected**: Ensure Azure OpenAI is properly configured
2. **ComfyUI Timeout**: Check ComfyUI server status and increase `max_attempts`
3. **Empty Image Paths**: Verify ComfyUI output directory permissions
4. **Memory Issues**: Reduce `images_per_second` for long videos

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

generator = RealtimeVideoGenerator()
# ... rest of code
```

## Demo Script

Run the included demo to see the system in action:

```bash
python realtime_demo.py
```

This demonstrates:
- Multiple content types
- Art style selection behavior
- Timestamp generation
- Prompt storage for ComfyUI
- State management

## Next Steps

1. Run `realtime_demo.py` to see the system in action
2. Integrate with your existing video generation pipeline
3. Customize art style categories in the prompts
4. Optimize `images_per_second` for your use case
5. Set up ComfyUI for actual image generation