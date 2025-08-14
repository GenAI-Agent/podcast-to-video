# Blur Background Video Generation

This document describes the new blur background video generation functionality that replicates and enhances the features from the Windows batch file `create_slideshow_blur_vertical_text.bat`.

## Overview

The new `create_video_with_blur_background()` method creates videos with a cinematic blur background effect, perfect for social media content, mobile viewing, and professional presentations.

## Key Features

### 🌫️ Blur Background Effect
- **Gaussian blur** with customizable strength (default: 24)
- Creates cinematic depth of field
- Original image remains sharp while background is artistically blurred
- Based on the FFmpeg filter chain from the Windows batch file

### 📐 Aspect Ratio Support
- **9:16 Vertical** (1080x1920) - Perfect for mobile, Instagram Stories, TikTok, YouTube Shorts
- **16:9 Horizontal** (1920x1080) - Traditional desktop/TV format
- Automatic dimension calculation based on selected ratio

### 🔤 Enhanced Typography
- **Arial Bold font** with black outline (5px border width)
- Text positioned **15% down from top** (matching batch file)
- White text with black outline for maximum visibility
- Centered horizontally on the frame

### 🎬 Professional Video Quality
- **30 FPS** frame rate
- **CRF 18** encoding (high quality)
- **H.264** video codec
- **AAC** audio encoding
- Enhanced subtitle styling for blur background visibility

## Usage Examples

### Basic Usage
```python
from src.generators.video_generator import VideoGenerator

generator = VideoGenerator()

# Create vertical video with blur background
output = generator.generate_video_from_article(
    article="Your content here...",
    output_path="blur_video.mp4",
    use_blur_background=True,
    overlay_text="Your Text",
    blur_strength=24,
    aspect_ratio="9:16"
)
```

### Advanced Usage
```python
# Create horizontal video with custom blur
output = generator.create_video_with_blur_background(
    image_paths=image_list,
    image_script=duration_script,
    audio_path="audio.mp3",
    srt_path="subtitles.srt",
    output_path="custom_blur.mp4",
    audio_duration=60.0,
    overlay_text="Custom Demo",
    blur_strength=36,  # Stronger blur
    aspect_ratio="16:9"  # Horizontal format
)
```

## Technical Implementation

### FFmpeg Filter Chain
The blur effect uses a complex filter chain that:

1. **Creates blurred background**: 
   - Scales image to fill frame
   - Crops to exact dimensions
   - Applies Gaussian blur

2. **Prepares main image**:
   - Scales to fit within frame
   - Maintains aspect ratio

3. **Composites layers**:
   - Overlays sharp image on blurred background
   - Centers the main image

4. **Adds text overlay**:
   - Uses Arial Bold font
   - Positions at 15% from top
   - Applies white color with black outline

### Filter Complex String
```bash
[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=24[blurred];
[0:v]scale=1080:1080:force_original_aspect_ratio=decrease[scaled];
[blurred][scaled]overlay=(W-w)/2:(H-h)/2[composed];
[composed]drawtext=text='Demo':fontsize=100:fontcolor=white:x=(w-text_w)/2:y=288:fontfile=/mnt/c/Windows/Fonts/arialbd.ttf:borderw=5:bordercolor=black[final]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_blur_background` | bool | False | Enable blur background effect |
| `overlay_text` | str | "Demo Video" | Text to display on video |
| `blur_strength` | int | 24 | Gaussian blur intensity |
| `aspect_ratio` | str | "9:16" | Video format ("9:16" or "16:9") |

## Blur Strength Guidelines

| Strength | Effect | Use Case |
|----------|--------|----------|
| 12-18 | Subtle blur | Professional presentations |
| 24 | Standard blur | Social media content |
| 30-36 | Strong blur | Artistic/creative content |
| 40+ | Extreme blur | Abstract/artistic effects |

## Compatibility

### Font Requirements
- **Windows**: Uses `C:/Windows/Fonts/arialbd.ttf`
- **WSL/Linux**: Uses `/mnt/c/Windows/Fonts/arialbd.ttf`
- Falls back to system fonts if Arial Bold not found

### Platform Support
- ✅ Windows with WSL
- ✅ Linux
- ✅ macOS (with font path adjustment)

## Output Formats

### Vertical (9:16) - Mobile Optimized
- **Resolution**: 1080x1920
- **Perfect for**: Instagram Stories, TikTok, YouTube Shorts, Snapchat
- **Text position**: 288px from top (15% of 1920px)

### Horizontal (16:9) - Desktop/TV
- **Resolution**: 1920x1080  
- **Perfect for**: YouTube, presentations, traditional media
- **Text position**: 162px from top (15% of 1080px)

## Demo Scripts

1. **`demo_blur_background.py`** - Comprehensive demo with multiple formats
2. **`src/examples/blur_background_example.py`** - Simple usage example

## Migration from Batch File

The Python implementation provides these advantages over the Windows batch file:

- ✅ Cross-platform compatibility
- ✅ Programmatic control
- ✅ Integration with existing video pipeline
- ✅ Advanced subtitle styling
- ✅ Automatic image search and selection
- ✅ Audio generation from text
- ✅ Error handling and fallbacks
- ✅ Temporary file cleanup

## Troubleshooting

### Common Issues

1. **Font not found**: Ensure Arial Bold is installed or adjust font path
2. **FFmpeg errors**: Check FFmpeg installation and version compatibility
3. **Image not found**: Verify image paths and file permissions
4. **Audio sync issues**: Check audio duration matches video length

### Debug Tips
- Enable verbose logging to see FFmpeg commands
- Check temporary directory for intermediate files
- Verify image list file format and paths
- Test with simple content first

## Future Enhancements

Potential improvements for future versions:
- Multiple blur zones/regions
- Animated blur effects
- Custom font support
- Dynamic text positioning
- Real-time preview
- Batch processing multiple videos