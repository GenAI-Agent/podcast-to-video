# Video Generation Guide

## 🎬 Quick Start

I've created several scripts to help you generate videos easily:

### Option 1: Quick Video Generator (Recommended)
```bash
python quick_video_generator.py
```

This interactive script offers:
- **Content Types**: AI Demo, Business, Technology, or Custom
- **Video Styles**: Horizontal blur, Vertical blur, or Standard
- **Custom Overlay Text**: Add your own text overlay

### Option 2: Demo Scripts
```bash
python demo_blur_background.py
```

### Option 3: Direct Video Generator
```bash
python src/generators/video_generator.py
```

## 🎥 Generated Video Features

Your videos will include:
- **AI-Generated Narration**: Text-to-speech using ElevenLabs
- **Smart Image Selection**: Vector search finds relevant images
- **Blur Background Effect**: Professional-looking background blur
- **Subtitle Overlay**: Auto-generated SRT subtitles
- **Custom Text Overlay**: Your branded text on the video
- **Multiple Aspect Ratios**: 16:9 (horizontal) or 9:16 (vertical)

## 📁 Existing Videos

I can see you already have several generated videos:
- `demo_artistic_blur.mp4` (4.8MB)
- `demo_horizontal_blur.mp4` (8.9MB) 
- `demo_vertical_blur.mp4` (5.4MB)
- `New_pre-market.mp4` (8.9MB)
- `pre-market.mp4` (9.1MB)
- `book_video.mp4` (9.1MB)
- `sample_video_generated_audio.mp4` (5.7MB)

## ⚙️ Prerequisites

Make sure you have:

1. **Environment Variables** (in `.env` file):
   ```
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_API_KEY=your_key
   ELEVENLABS_API_KEY=your_key
   PINECONE_API_KEY=your_key
   ```

2. **FFmpeg** installed and in your PATH

3. **Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎨 Video Customization Options

### Blur Background Effect
- **Strength**: Adjustable blur intensity (0-50)
- **Aspect Ratios**: 16:9 (horizontal) or 9:16 (vertical)
- **Text Positioning**: Automatically positioned 15% from top

### Content Options
- **AI Demo**: Technology and AI content
- **Business**: Corporate and business topics  
- **Technology**: Tech industry focus
- **Custom**: Your own content

### Audio Generation
- **Text-to-Speech**: High-quality AI voices
- **Custom Audio**: Use your own audio files
- **Subtitle Sync**: Auto-synchronized subtitles

## 🚀 Example Usage

### Quick Generation
```python
from quick_video_generator import generate_video_quick

# Generate AI demo with horizontal blur
video_path = generate_video_quick(
    content_type="ai_demo",
    video_style="blur_horizontal", 
    overlay_text="My AI Demo"
)
```

### Advanced Generation
```python
from src.generators.video_generator import VideoGenerator

generator = VideoGenerator()
video = generator.generate_video_from_article(
    article="Your content here...",
    output_path="my_video.mp4",
    use_gpt_transcript=True,
    use_blur_background=True,
    overlay_text="Custom Text",
    aspect_ratio="16:9"
)
```

## 🛠️ Troubleshooting

### Common Issues:
1. **FFmpeg not found**: Install FFmpeg and add to PATH
2. **API key errors**: Check your `.env` file
3. **Memory issues**: Reduce video length or image count
4. **Network errors**: Check internet connection for API calls

### Getting Help:
- Check the console output for detailed error messages
- Verify all dependencies are installed
- Ensure API keys are valid and have sufficient credits

## 📊 Performance Tips

- **Shorter content** = faster generation
- **Fewer images** = smaller file sizes
- **Lower blur strength** = faster processing
- **Standard format** vs blur = quicker generation

---

**Ready to create your video?** Run `python quick_video_generator.py` and follow the prompts!