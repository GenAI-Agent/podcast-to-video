# Video Generation API Integration Guide

## Overview
This guide explains how to use the integrated video generation system that connects your frontend at http://localhost:3001/upload to the video_generator.py backend via Flask API.

## Architecture

```
Frontend (Next.js)          Backend (Flask)           Video Generator
localhost:3001/upload  -->  localhost:5000/api  -->  video_generator.py
```

## Setup Instructions

### 1. Install Dependencies

#### Backend (Python)
```bash
pip install flask flask-cors
```

#### Frontend (Next.js)
```bash
cd frontend
npm install
```

### 2. Start the Servers

#### Option A: Use the startup script (Recommended)
```bash
./start_servers.sh
```

#### Option B: Start manually

Terminal 1 - Backend:
```bash
python3 api_server.py
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

## API Endpoints

### 1. Generate Video
**Endpoint:** `POST http://172.25.27.208:5000/api/generate-video`

**Form Data Parameters:**
- `article_text` (string): Text content for video generation
- `web_link` (string): URL to extract content from (optional)
- `use_gpt_transcript` (boolean): Whether to generate transcript using GPT
- `custom_audio` (file): Custom audio file (MP3/WAV) (optional)
- `custom_srt` (file): Custom subtitle file (SRT) (optional)
- `use_blur_background` (boolean): Enable blur background effect
- `overlay_text` (string): Text to overlay on video
- `blur_strength` (integer): Blur strength (1-50, default: 24)
- `aspect_ratio` (string): "9:16" (vertical) or "16:9" (horizontal)

**Response:**
```json
{
  "success": true,
  "message": "Video generated successfully",
  "video_url": "/api/download/video_20250811_123456.mp4",
  "filename": "video_20250811_123456.mp4"
}
```

### 2. Generate Audio Only
**Endpoint:** `POST http://172.25.27.208:5000/api/generate-audio`

**JSON Body:**
```json
{
  "text": "Your text content here"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Audio generated successfully",
  "audio_url": "/api/download/audio_20250811_123456.mp3",
  "filename": "audio_20250811_123456.mp3",
  "duration": 45.2
}
```

### 3. Download File
**Endpoint:** `GET http://172.25.27.208:5000/api/download/<filename>`

Downloads the generated video or audio file.

### 4. List Videos
**Endpoint:** `GET http://172.25.27.208:5000/api/list-videos`

Returns a list of all generated videos.

## Frontend Usage

### Access the Upload Page
Navigate to: http://localhost:3001/upload

### Features Available:

1. **Content Input**
   - Enter article text directly
   - OR provide a web link to extract content
   - OR upload custom SRT file

2. **Optional Files**
   - Upload custom audio (MP3/WAV)
   - Upload custom subtitles (SRT)

3. **Video Options**
   - Use GPT Transcript: Generate a 1-minute transcript from article
   - Blur Background Effect: Add artistic blur background
   - Overlay Text: Add text overlay on video
   - Blur Strength: Adjust blur intensity (1-50)
   - Aspect Ratio: Choose vertical (9:16) or horizontal (16:9)

4. **Actions**
   - Generate Video: Creates full video with all options
   - Generate Audio Only: Creates just the audio from text

## Example Usage

### Basic Video Generation
1. Open http://localhost:3001/upload
2. Enter your article text
3. Click "Generate Video"
4. Wait for processing (may take 1-2 minutes)
5. Preview and download the generated video

### Advanced Video with Effects
1. Enter article text or web link
2. Check "Use GPT Transcript" for AI-generated script
3. Check "Blur Background Effect"
4. Set overlay text (e.g., "My Brand")
5. Adjust blur strength slider
6. Choose aspect ratio (9:16 for social media, 16:9 for YouTube)
7. Click "Generate Video"

### Audio-Only Generation
1. Enter your text
2. Click "Generate Audio Only"
3. Download the generated MP3 file

## Environment Variables Required

Create a `.env` file in the project root with:

```env
# Azure OpenAI (for GPT transcript generation)
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key

# ElevenLabs (for audio generation)
ELEVENLABS_API_KEY=your_api_key

# Pinecone (for image search)
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=your_environment
```

## File Structure

```
batch_image/
├── api_server.py           # Flask API server
├── frontend/
│   └── src/app/upload/     # Upload page UI
├── src/generators/
│   └── video_generator.py  # Core video generation logic
├── uploads/                # Temporary uploaded files
├── generated_videos/       # Generated video/audio files
└── start_servers.sh        # Startup script
```

## Troubleshooting

### CORS Issues
The Flask server is configured to accept requests from localhost:3001. If you're running the frontend on a different port, update the CORS configuration in `api_server.py`.

### File Upload Limits
Default file upload limits may apply. For large files, you may need to adjust:
- Next.js: Update `next.config.js`
- Flask: Set `MAX_CONTENT_LENGTH` in `api_server.py`

### Missing Dependencies
If you encounter import errors, ensure all Python dependencies are installed:
```bash
pip install -r requirements.txt
```

### Port Conflicts
If ports 3001 or 5000 are already in use:
- Frontend: Change port in `frontend/package.json` dev script
- Backend: Change port in `api_server.py` last line

## API Testing with curl

Test video generation:
```bash
curl -X POST http://172.25.27.208:5000/api/generate-video \
  -F "article_text=Your article content here" \
  -F "use_gpt_transcript=true" \
  -F "aspect_ratio=16:9"
```

Test audio generation:
```bash
curl -X POST http://172.25.27.208:5000/api/generate-audio \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

## Security Notes

- The API server runs on localhost only by default
- For production deployment, add proper authentication
- Sanitize all user inputs
- Implement rate limiting for API endpoints
- Use HTTPS in production

## Support

For issues or questions:
1. Check the console logs in both terminal windows
2. Verify all environment variables are set
3. Ensure all dependencies are installed
4. Check file permissions in upload/output directories