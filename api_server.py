#!/usr/bin/env python3
"""
Flask API server for video generation
Provides endpoints for the frontend to generate videos
"""

import os
import sys
import json
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.generators.video_generator import VideoGenerator
from src.generators.realtime_adapter import RealtimeVideoGeneratorAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=['http://localhost:3001', 'http://localhost:3000', 'http://localhost:3002', 'http://172.25.27.208:3001', 'http://172.25.27.208:3000', 'http://172.25.27.208:3002'])

# Configuration
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')
OUTPUT_FOLDER = os.path.join(project_root, 'generated_videos')
ALLOWED_EXTENSIONS = {'txt', 'json', 'mp3', 'wav', 'srt'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_generator(backend_type='standard', restricted_json_file=None, topic=None):
    """Factory function to get the appropriate video generator based on backend type"""
    if backend_type == 'realtime':
        logger.info("Using Realtime Video Generator (via adapter)")
        # Get ComfyUI URL from environment
        comfyui_url = os.getenv("COMFYUI_URL", "https://7fd6781ec07e.ngrok-free.app/api/prompt")
        # Use the adapter to make RealtimeVideoGenerator compatible
        return RealtimeVideoGeneratorAdapter(comfyui_url=comfyui_url)
    else:
        logger.info("Using Standard Video Generator")
        return VideoGenerator(restricted_json_file=restricted_json_file, topic=topic)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Video generation API is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/prepare-video', methods=['POST'])
def prepare_video():
    """
    Prepare video content (script, audio, and images) without creating final video
    
    Expected JSON data:
    - text: Text content or web link for video generation
    - topic: The topic for image filtering (optional)
    - use_gpt_transcript: Whether to generate transcript using GPT (boolean)
    - backend_type: 'standard' or 'realtime' (optional, defaults to 'standard')
    - use_comfyui: Whether to use ComfyUI for image generation (boolean, only for realtime backend)
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        topic = data.get('topic', 'general')
        use_gpt_transcript = data.get('use_gpt_transcript', False)
        backend_type = data.get('backend_type', 'standard')
        use_comfyui = data.get('use_comfyui', False)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        logger.info(f"Preparing video content using {backend_type} backend")
        logger.info(f"Text length: {len(text)}, Topic: {topic}")
        
        # Set up topic filtering
        topic_json_map = {
            'astrology': 'data/json/batch_image_data_20250801_100459_with_tags_updated.json',
            'trading': 'data/json/batch_image_data_20250801_105709_with_tags_updated.json',
            'fantasy': 'data/json/fantasy_image_data_20250805_091354_final.json',
            'horror': 'data/json/spooky_story_data_20250805_113814_final.json',
            'romance': 'data/json/romance_data_20250805_133414_final.json',
            'drama': 'data/json/drama_data_20250805_144158_final.json',
            'thriller': 'data/json/batch_image_data_20250801_133312_with_tags_updated.json'
        }
        
        restricted_json_file = None
        if topic and topic in topic_json_map:
            restricted_json_file = os.path.join(project_root, topic_json_map[topic])
            if os.path.exists(restricted_json_file):
                logger.info(f"Using topic-specific image set: {restricted_json_file}")
            else:
                logger.warning(f"Topic JSON file not found: {restricted_json_file}")
                restricted_json_file = None
        
        # Initialize the appropriate generator
        generator = get_generator(backend_type, restricted_json_file, topic)
        
        # Handle web link extraction if needed
        article_text = text
        if text.startswith('http'):
            logger.info("Extracting content from web link...")
            article_text = generator.extract_web_content(text)
        
        # Generate transcript if requested
        transcript = article_text
        if use_gpt_transcript and article_text:
            logger.info("Generating transcript using GPT...")
            transcript = generator.generate_transcript_from_article(article_text)
        
        # Generate audio
        logger.info("Generating audio...")
        audio_path, duration = generator.generate_audio_from_api(transcript)
        
        # Move audio to output folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"audio_{timestamp}.mp3"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        import shutil
        shutil.move(audio_path, output_path)
        
        # Generate image script and search/generate images
        logger.info("Generating image script and processing images...")
        image_script = generator.get_image_list_script(transcript, duration)
        
        if image_script:
            descriptions = [item["description"] for item in image_script]
        else:
            # Fallback to sentences
            import re
            descriptions = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
        
        # Handle image/video generation based on backend and ComfyUI option
        if backend_type == 'realtime' and use_comfyui:
            logger.info("Using ComfyUI for video/image generation...")
            # Get art style for all content
            art_style_result = generator.realtime_gen.process_text_to_image_prompt(transcript)
            art_style = art_style_result.get('art_style', 'photorealistic')
            
            # Try direct video generation first using IMAGE_PROMPT system
            logger.info("Attempting ComfyUI video generation using IMAGE_PROMPT...")
            video_path = generator.realtime_gen.generate_video_with_comfyui(transcript, art_style, int(duration))
            
            if video_path and os.path.exists(video_path):
                logger.info(f"✅ ComfyUI generated complete video: {video_path}")
                # For complete video, we return it as a single item
                image_paths = [video_path]
                descriptions = ["Complete video generated by ComfyUI using IMAGE_PROMPT system"]
            else:
                logger.info("ComfyUI video generation failed, falling back to single image generation...")
                # Generate a single comprehensive image with ComfyUI
                single_image = generator.realtime_gen.generate_single_image_with_comfyui(transcript, art_style)
                if single_image:
                    # Use the same image for all segments
                    image_paths = [single_image] * len(descriptions)
                    logger.info(f"Generated 1 comprehensive image with ComfyUI, using for {len(descriptions)} segments")
                else:
                    logger.info("Single image generation failed, using database fallback...")
                    image_paths = generator.vector_search_images(descriptions)
        else:
            # Search for existing images in database
            logger.info("Searching for existing images in database...")
            image_paths = generator.vector_search_images(descriptions)
        
        # Collect image previews
        image_previews = generator._collect_image_previews(image_paths, descriptions)
        
        total_found = len([p for p in image_paths if p])
        
        logger.info(f"Video preparation completed successfully")
        logger.info(f"- Transcript: {len(transcript)} characters")
        logger.info(f"- Audio: {duration:.2f} seconds")
        logger.info(f"- Images: {total_found}/{len(descriptions)} found")
        
        return jsonify({
            'success': True,
            'message': 'Video content prepared successfully',
            'transcript': transcript,
            'audio_url': f'/api/download/{output_filename}',
            'duration': duration,
            'image_previews': image_previews,
            'image_paths': image_paths,
            'descriptions': descriptions,
            'total_images': total_found,
            'total_segments': len(descriptions),
            'backend_used': backend_type
        })
        
    except Exception as e:
        logger.error(f"Error preparing video: {str(e)}")
        return jsonify({
            'error': 'Failed to prepare video',
            'details': str(e)
        }), 500

@app.route('/api/create-final-video', methods=['POST'])
def create_final_video():
    """
    Create final video from prepared content
    
    Expected form data:
    - transcript: The prepared transcript text
    - audio_filename: Filename of the prepared audio file
    - topic: The topic for image filtering (optional)
    - use_blur_background: Whether to use blur background effect (boolean)
    - overlay_text: Text to overlay on video (optional)
    - blur_strength: Blur strength (integer, default 24)
    - aspect_ratio: Video aspect ratio ("9:16" or "16:9")
    - backend_type: 'standard' or 'realtime' (optional, defaults to 'standard')
    - image_paths: JSON array of prepared image paths (optional)
    - descriptions: JSON array of prepared descriptions (optional)
    - custom_audio: Custom audio file (optional)
    - custom_srt: Custom SRT file (optional)
    """
    try:
        # Parse request data
        transcript = request.form.get('transcript', '')
        audio_filename = request.form.get('audio_filename', '')
        topic = request.form.get('topic', 'general')
        use_blur_background = request.form.get('use_blur_background', 'false').lower() == 'true'
        overlay_text = request.form.get('overlay_text', 'Demo Video')
        blur_strength = int(request.form.get('blur_strength', '24'))
        aspect_ratio = request.form.get('aspect_ratio', '9:16')
        backend_type = request.form.get('backend_type', 'standard')
        
        logger.info(f"Creating final video using {backend_type} backend")
        
        # Handle prepared image data
        prepared_image_paths = None
        prepared_descriptions = None
        try:
            image_paths_json = request.form.get('image_paths')
            descriptions_json = request.form.get('descriptions')
            if image_paths_json and descriptions_json:
                prepared_image_paths = json.loads(image_paths_json)
                prepared_descriptions = json.loads(descriptions_json)
                logger.info(f"Using prepared image data: {len(prepared_image_paths)} images")
        except:
            logger.info("No prepared image data provided, will search for images")
        
        # Handle custom files
        custom_audio_path = None
        custom_srt_path = None
        
        if 'custom_audio' in request.files:
            audio_file = request.files['custom_audio']
            if audio_file and allowed_file(audio_file.filename):
                audio_filename_custom = secure_filename(audio_file.filename)
                custom_audio_path = os.path.join(UPLOAD_FOLDER, f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{audio_filename_custom}")
                audio_file.save(custom_audio_path)
                logger.info(f"Saved custom audio: {custom_audio_path}")
        
        if 'custom_srt' in request.files:
            srt_file = request.files['custom_srt']
            if srt_file and allowed_file(srt_file.filename):
                srt_filename = secure_filename(srt_file.filename)
                custom_srt_path = os.path.join(UPLOAD_FOLDER, f"srt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{srt_filename}")
                srt_file.save(custom_srt_path)
                logger.info(f"Saved custom SRT: {custom_srt_path}")
        
        # Validate input
        if not transcript and not custom_srt_path:
            return jsonify({'error': 'No transcript or custom SRT provided'}), 400
        
        if not audio_filename and not custom_audio_path:
            return jsonify({'error': 'No audio file provided'}), 400
        
        # Set up topic filtering
        topic_json_map = {
            'astrology': 'data/json/batch_image_data_20250801_100459_with_tags_updated.json',
            'trading': 'data/json/batch_image_data_20250801_105709_with_tags_updated.json',
            'fantasy': 'data/json/fantasy_image_data_20250805_091354_final.json',
            'horror': 'data/json/spooky_story_data_20250805_113814_final.json',
            'romance': 'data/json/romance_data_20250805_133414_final.json',
            'drama': 'data/json/drama_data_20250805_144158_final.json',
            'thriller': 'data/json/batch_image_data_20250801_133312_with_tags_updated.json'
        }
        
        restricted_json_file = None
        if topic and topic in topic_json_map:
            restricted_json_file = os.path.join(project_root, topic_json_map[topic])
            if os.path.exists(restricted_json_file):
                logger.info(f"Using topic-specific image set: {restricted_json_file}")
            else:
                logger.warning(f"Topic JSON file not found: {restricted_json_file}")
                restricted_json_file = None
        
        # Initialize the appropriate generator
        generator = get_generator(backend_type, restricted_json_file, topic)
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        topic_suffix = f"_{topic}" if topic else ""
        backend_suffix = f"_{backend_type}" if backend_type != 'standard' else ""
        output_filename = f"video{topic_suffix}{backend_suffix}_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Get prepared audio path
        prepared_audio_path = custom_audio_path if custom_audio_path else os.path.join(OUTPUT_FOLDER, audio_filename)
        
        if not os.path.exists(prepared_audio_path):
            return jsonify({'error': f'Audio file not found: {audio_filename}'}), 400
        
        # Create video using the appropriate method
        if prepared_image_paths and prepared_descriptions:
            logger.info("Creating video with prepared image assignments...")
            video_path = generator.create_video_from_prepared_data_with_images(
                transcript=transcript,
                audio_path=prepared_audio_path,
                output_path=output_path,
                prepared_image_paths=prepared_image_paths,
                prepared_descriptions=prepared_descriptions,
                custom_srt_path=custom_srt_path,
                use_blur_background=use_blur_background,
                overlay_text=overlay_text,
                blur_strength=blur_strength,
                aspect_ratio=aspect_ratio
            )
        else:
            logger.info("Creating video from prepared data (will search for images)...")
            video_path = generator.create_video_from_prepared_data(
                transcript=transcript,
                audio_path=prepared_audio_path,
                output_path=output_path,
                custom_srt_path=custom_srt_path,
                use_blur_background=use_blur_background,
                overlay_text=overlay_text,
                blur_strength=blur_strength,
                aspect_ratio=aspect_ratio
            )
        
        logger.info(f"Final video created successfully: {video_path}")
        
        # Clean up temporary files
        if custom_audio_path and os.path.exists(custom_audio_path):
            os.remove(custom_audio_path)
        if custom_srt_path and os.path.exists(custom_srt_path):
            os.remove(custom_srt_path)
        
        return jsonify({
            'success': True,
            'message': 'Final video created successfully',
            'video_url': f'/api/download/{output_filename}',
            'filename': output_filename,
            'backend_used': backend_type
        })
        
    except Exception as e:
        logger.error(f"Error creating final video: {str(e)}")
        return jsonify({
            'error': 'Failed to create final video',
            'details': str(e)
        }), 500

@app.route('/api/generate-video', methods=['POST'])
def generate_video():
    """
    Generate video from uploaded content
    
    Expected form data:
    - article_text: Text content for video generation
    - web_link: URL to extract content from (optional)
    - use_gpt_transcript: Whether to generate transcript using GPT (boolean)
    - custom_audio: Audio file (optional)
    - custom_srt: SRT file (optional)
    - use_blur_background: Whether to use blur background effect (boolean)
    - overlay_text: Text to overlay on video (optional)
    - blur_strength: Blur strength (integer, default 24)
    - aspect_ratio: Video aspect ratio ("9:16" or "16:9")
    """
    try:
        logger.info("Received video generation request")
        
        # Parse request data
        article_text = request.form.get('article_text', '')
        web_link = request.form.get('web_link', '')
        topic = request.form.get('topic', '')
        use_gpt_transcript = request.form.get('use_gpt_transcript', 'false').lower() == 'true'
        use_blur_background = request.form.get('use_blur_background', 'false').lower() == 'true'
        overlay_text = request.form.get('overlay_text', 'Demo Video')
        blur_strength = int(request.form.get('blur_strength', '24'))
        aspect_ratio = request.form.get('aspect_ratio', '9:16')
        backend_type = request.form.get('backend_type', 'standard')
        
        # Log the selected topic
        if topic:
            logger.info(f"Selected topic: {topic}")
        
        # Handle file uploads
        custom_audio_path = None
        custom_srt_path = None
        
        if 'custom_audio' in request.files:
            audio_file = request.files['custom_audio']
            if audio_file and allowed_file(audio_file.filename):
                audio_filename = secure_filename(audio_file.filename)
                custom_audio_path = os.path.join(UPLOAD_FOLDER, f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{audio_filename}")
                audio_file.save(custom_audio_path)
                logger.info(f"Saved custom audio: {custom_audio_path}")
        
        if 'custom_srt' in request.files:
            srt_file = request.files['custom_srt']
            if srt_file and allowed_file(srt_file.filename):
                srt_filename = secure_filename(srt_file.filename)
                custom_srt_path = os.path.join(UPLOAD_FOLDER, f"srt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{srt_filename}")
                srt_file.save(custom_srt_path)
                logger.info(f"Saved custom SRT: {custom_srt_path}")
        
        # Validate input
        if not article_text and not web_link and not custom_srt_path:
            return jsonify({
                'error': 'Please provide article text, web link, or custom SRT file'
            }), 400
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        topic_suffix = f"_{topic}" if topic else ""
        output_filename = f"video{topic_suffix}_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Initialize video generator with topic-specific JSON if available
        logger.info(f"Initializing video generator using {backend_type} backend...")
        
        # Map topics to their JSON files
        topic_json_map = {
            'astrology': 'data/json/batch_image_data_20250801_100459_with_tags_updated.json',
            'trading': 'data/json/batch_image_data_20250801_105709_with_tags_updated.json',
            'fantasy': 'data/json/fantasy_image_data_20250805_091354_final.json',
            'horror': 'data/json/spooky_story_data_20250805_113814_final.json',
            'romance': 'data/json/romance_data_20250805_133414_final.json',
            'drama': 'data/json/drama_data_20250805_144158_final.json',
            'thriller': 'data/json/batch_image_data_20250801_133312_with_tags_updated.json'
        }
        
        restricted_json_file = None
        if topic and topic in topic_json_map:
            restricted_json_file = os.path.join(project_root, topic_json_map[topic])
            if os.path.exists(restricted_json_file):
                logger.info(f"Using topic-specific image set: {restricted_json_file}")
            else:
                logger.warning(f"Topic JSON file not found: {restricted_json_file}")
                restricted_json_file = None
        
        generator = get_generator(backend_type, restricted_json_file, topic)
        
        # Generate transcript first if needed (to return it in response)
        transcript = article_text
        if use_gpt_transcript and article_text:
            logger.info("Generating transcript using GPT...")
            transcript = generator.generate_transcript_from_article(article_text)
        
        # Generate video
        logger.info("Starting video generation...")
        result = generator.generate_video_from_article(
            article=transcript if use_gpt_transcript else article_text if article_text else None,
            web_link=web_link if web_link else None,
            output_path=output_path,
            use_gpt_transcript=False,  # We already generated it above
            custom_srt_path=custom_srt_path,
            custom_audio_path=custom_audio_path,
            use_blur_background=use_blur_background,
            overlay_text=overlay_text,
            blur_strength=blur_strength,
            aspect_ratio=aspect_ratio
        )
        
        # Handle both old string return format and new dict format
        if isinstance(result, str):
            # Old format - just video path
            result_path = result
            image_previews = []
            total_images = 0
        else:
            # New format - dict with video_path and image_previews
            result_path = result["video_path"]
            image_previews = result.get("image_previews", [])
            total_images = result.get("total_images", 0)
        
        logger.info(f"Video generated successfully: {result_path}")
        logger.info(f"Generated with {total_images} images")
        
        # Clean up temporary files
        if custom_audio_path and os.path.exists(custom_audio_path):
            os.remove(custom_audio_path)
        if custom_srt_path and os.path.exists(custom_srt_path):
            os.remove(custom_srt_path)
        
        # Return success response with video URL, transcript, and image previews
        return jsonify({
            'success': True,
            'message': 'Video generated successfully',
            'video_url': f'/api/download/{output_filename}',
            'filename': output_filename,
            'transcript': transcript if (use_gpt_transcript or article_text) else None,
            'transcript_generated': use_gpt_transcript,
            'image_previews': image_previews,
            'total_images': total_images
        })
        
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        return jsonify({
            'error': 'Failed to generate video',
            'details': str(e)
        }), 500

@app.route('/api/generate-audio', methods=['POST'])
def generate_audio():
    """
    Generate audio from text using ElevenLabs API
    
    Expected JSON data:
    - text: Text content to convert to audio
    - use_gpt_transcript: Whether to generate transcript using GPT (optional)
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        use_gpt_transcript = data.get('use_gpt_transcript', False)
        backend_type = data.get('backend_type', 'standard')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        logger.info(f"Generating audio for text of length: {len(text)}")
        
        # Initialize video generator to use its audio generation method
        generator = get_generator(backend_type)
        
        # Generate transcript if requested
        transcript = text
        if use_gpt_transcript:
            logger.info("Generating transcript using GPT...")
            transcript = generator.generate_transcript_from_article(text)
            logger.info(f"Generated transcript of length: {len(transcript)}")
        
        # Generate audio using the transcript
        audio_path, duration = generator.generate_audio_from_api(transcript)
        
        # Move audio to output folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"audio_{timestamp}.mp3"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        import shutil
        shutil.move(audio_path, output_path)
        
        logger.info(f"Audio generated successfully: {output_path}, duration: {duration}s")
        
        return jsonify({
            'success': True,
            'message': 'Audio generated successfully',
            'audio_url': f'/api/download/{output_filename}',
            'filename': output_filename,
            'duration': duration,
            'transcript': transcript,
            'transcript_generated': use_gpt_transcript
        })
        
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return jsonify({
            'error': 'Failed to generate audio',
            'details': str(e)
        }), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download generated video or audio file"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Determine MIME type based on file extension
        if filename.endswith('.mp4'):
            mimetype = 'video/mp4'
        elif filename.endswith('.mp3'):
            mimetype = 'audio/mpeg'
        elif filename.endswith('.wav'):
            mimetype = 'audio/wav'
        else:
            mimetype = 'application/octet-stream'
        
        return send_file(
            file_path,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({
            'error': 'Failed to download file',
            'details': str(e)
        }), 500

@app.route('/api/image-preview/<path:image_path>', methods=['GET'])
def serve_image_preview(image_path):
    """Serve image preview files"""
    try:
        # Security: Only allow specific image directories
        allowed_dirs = [
            '/home/fluxmind/batch_image/data',
            '/mnt/c/Users/x7048/Documents/ComfyUI/output',
            'C:/Users/x7048/Documents/ComfyUI/output'
        ]
        
        # Find the image file in allowed directories
        full_path = None
        for base_dir in allowed_dirs:
            if base_dir.startswith('C:'):
                # Windows path - convert to WSL
                wsl_base = base_dir.replace('C:', '/mnt/c').replace('\\', '/')
            else:
                wsl_base = base_dir
                
            # Try different subdirectories
            potential_paths = [
                os.path.join(wsl_base, image_path),
                os.path.join(wsl_base, 'trading', image_path),
                os.path.join(wsl_base, 'fantasy_adventure', image_path),
                os.path.join(wsl_base, 'day_trading', image_path),
            ]
            
            for potential_path in potential_paths:
                if os.path.exists(potential_path) and os.path.isfile(potential_path):
                    full_path = potential_path
                    break
            
            if full_path:
                break
        
        if not full_path or not os.path.exists(full_path):
            return jsonify({'error': 'Image not found'}), 404
        
        # Security check: ensure the resolved path is within allowed directories
        abs_path = os.path.abspath(full_path)
        is_allowed = any(abs_path.startswith(os.path.abspath(d.replace('C:', '/mnt/c').replace('\\', '/'))) 
                        for d in allowed_dirs)
        
        if not is_allowed:
            return jsonify({'error': 'Access denied'}), 403
        
        return send_file(abs_path, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Error serving image preview: {str(e)}")
        return jsonify({
            'error': 'Failed to serve image',
            'details': str(e)
        }), 500

@app.route('/api/assign-images', methods=['POST'])
def assign_images():
    """
    Assign images to script segments without generating video
    
    Expected JSON data:
    - transcript: The script text to segment and find images for
    - topic: The topic for image filtering (optional)
    """
    try:
        data = request.get_json()
        transcript = data.get('transcript', '')
        topic = data.get('topic', '')
        backend_type = data.get('backend_type', 'standard')
        
        if not transcript:
            return jsonify({'error': 'No transcript provided'}), 400
        
        logger.info(f"Assigning images for transcript (length: {len(transcript)})")
        
        # Set up topic filtering
        restricted_json_file = None
        topic_json_map = {
            'trading': 'data/json/batch_image_data_20250801_105709_with_tags_updated.json',
            'fantasy': 'data/json/fantasy_image_data_20250805_091354_final.json',
            'astrology': 'data/json/astrology_image_data_final.json',
            'romance': 'data/json/romance_image_data_final.json',
            'horror': 'data/json/horror_image_data_final.json', 
            'drama': 'data/json/drama_image_data_final.json',
            'thriller': 'data/json/thriller_image_data_final.json'
        }
        
        if topic and topic in topic_json_map:
            project_root = os.path.dirname(os.path.abspath(__file__))
            restricted_json_file = os.path.join(project_root, topic_json_map[topic])
            if os.path.exists(restricted_json_file):
                logger.info(f"Using topic-specific image set: {restricted_json_file}")
            else:
                logger.warning(f"Topic JSON file not found: {restricted_json_file}")
                restricted_json_file = None
        
        # Initialize video generator for image search
        generator = get_generator(backend_type, restricted_json_file, topic)
        
        # Generate image script (descriptions for each segment)
        import re
        sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
        
        # Try to generate structured image descriptions
        try:
            # Estimate audio duration (rough estimate: 150 words per minute, 5 chars per word average)
            estimated_duration = len(transcript) / 5 / 150 * 60
            image_script = generator.get_image_list_script(transcript, estimated_duration)
            if image_script:
                descriptions = [item["description"] for item in image_script]
                logger.info(f"Generated {len(descriptions)} image descriptions")
            else:
                # Fallback to sentences
                descriptions = sentences
                logger.info(f"Using {len(descriptions)} sentences as descriptions")
        except Exception as e:
            logger.warning(f"Image script generation failed, using sentences: {e}")
            descriptions = sentences
        
        # Search for images
        logger.info("Searching for images...")
        image_paths = generator.vector_search_images(descriptions)
        
        # Collect image previews
        image_previews = generator._collect_image_previews(image_paths, descriptions)
        
        total_found = len([p for p in image_paths if p])
        logger.info(f"Image assignment completed: {total_found}/{len(descriptions)} images found")
        
        return jsonify({
            'success': True,
            'message': f'Images assigned successfully! Found {total_found} of {len(descriptions)} images.',
            'image_previews': image_previews,
            'total_images': total_found,
            'total_segments': len(descriptions)
        })
        
    except Exception as e:
        logger.error(f"Error assigning images: {str(e)}")
        return jsonify({
            'error': 'Failed to assign images',
            'details': str(e)
        }), 500

@app.route('/api/list-videos', methods=['GET'])
def list_videos():
    """List all generated videos"""
    try:
        videos = []
        for filename in os.listdir(OUTPUT_FOLDER):
            if filename.endswith('.mp4'):
                file_path = os.path.join(OUTPUT_FOLDER, filename)
                file_stat = os.stat(file_path)
                videos.append({
                    'filename': filename,
                    'size': file_stat.st_size,
                    'created': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                    'url': f'/api/download/{filename}'
                })
        
        # Sort by creation time, newest first
        videos.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'videos': videos,
            'count': len(videos)
        })
        
    except Exception as e:
        logger.error(f"Error listing videos: {str(e)}")
        return jsonify({
            'error': 'Failed to list videos',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting Flask API server...")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Output folder: {OUTPUT_FOLDER}")
    app.run(host='0.0.0.0', port=5000, debug=True)