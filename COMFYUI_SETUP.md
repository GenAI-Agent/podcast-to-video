# ComfyUI Integration Setup Guide

## ✅ Integration Complete

The RealtimeVideoGenerator now supports ComfyUI integration for custom image generation!

## 🔧 Setup Steps

### 1. Update ComfyUI URL
Set your ComfyUI endpoint URL in one of these ways:

**Option A: Environment Variable**
```bash
export COMFYUI_URL="https://your-ngrok-url.app/api/prompt"
```

**Option B: Update in code**
- Edit `image_generator.py` line 82: `COMFY_URL = "your-ngrok-url"`
- Edit API server default in `api_server.py` line 49

### 2. Frontend Usage

1. Go to the upload page
2. Select "Realtime Generator" backend
3. Check "🎨 Use ComfyUI for Image Generation"
4. Upload your article/text
5. Click "Generate Script & Audio"

### 3. API Usage

```python
# Using the integrated pipeline
from src.generators.Realtime_Video_Gen import RealtimeVideoGenerator

generator = RealtimeVideoGenerator(comfyui_url="your-comfyui-url")

video_path = generator.process_article_to_video(
    article="Your article text here",
    output_path="output.mp4", 
    target_duration=30,
    use_comfyui=True,        # Enable ComfyUI generation
    aspect_ratio="9:16"
)
```

### 4. API Endpoint
```bash
POST /api/prepare-video
Content-Type: application/json

{
  "text": "Your article text",
  "backend_type": "realtime",
  "use_comfyui": true,
  "use_gpt_transcript": true
}
```

## 🎯 Pipeline Flow

**NEW: Direct Video Generation (Primary)**
```
Article → Art Style Selection → IMAGE_PROMPT Generation → 
ComfyUI Video Generation → Complete Video Output
```

**Fallback: Image-Based Compilation**
```
Article → Art Style Selection → Scene Breakdown → 
ComfyUI Image Generation → Audio Generation → 
Subtitle Generation → Video Compilation
```

## 🔧 Key Features

- **🎬 Direct Video Generation**: ComfyUI generates complete videos using IMAGE_PROMPT system
- **🎨 Automatic Art Style Selection**: AI chooses the best visual style for your content
- **📝 IMAGE_PROMPT System**: Uses comprehensive prompts for entire video content
- **🔄 Smart Fallback**: Falls back to image-based compilation if direct video fails
- **📋 Scene Breakdown**: Intelligently divides content into visual scenes (fallback mode)
- **🖼️ ComfyUI Integration**: Supports both video and image generation workflows
- **🛡️ Robust Fallback**: Uses database images as final fallback
- **🎵 Full Pipeline**: Handles audio, subtitles, and final compilation

## 📋 Testing

Run the test script:
```bash
python test_comfyui_integration.py
```

## 🐛 Troubleshooting

### Common Issues:

1. **ComfyUI Connection Failed**
   - Check your ngrok URL is active
   - Verify the URL ends with `/api/prompt`
   - Test ComfyUI manually first

2. **Image Generation Timeout**
   - Adjust wait time in `generate_images_with_comfyui()` method
   - Check ComfyUI processing queue

3. **Missing Dependencies**
   - Ensure all Python packages are installed
   - Check Azure OpenAI and ElevenLabs API keys

### Expected Image Paths:
ComfyUI images are expected at:
```
/mnt/c/Users/x7048/Documents/ComfyUI/output/{folder_name}/
```

Adjust the path in the `generate_images_with_comfyui()` method if different.

## 🎉 What's New

- ✅ Complete ComfyUI integration
- ✅ Dual art style prompting system  
- ✅ Intelligent scene breakdown
- ✅ Frontend ComfyUI toggle
- ✅ API endpoint support
- ✅ Fallback mechanisms
- ✅ Full video pipeline

The integration is ready to use! 🚀