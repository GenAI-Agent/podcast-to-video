#!/usr/bin/env python3
"""
Debug the entire subtitle pipeline to find where subtitles are getting lost
"""
import os
import sys
import subprocess
import tempfile

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.generators.video_generator import VideoGenerator

def debug_subtitle_pipeline():
    """Debug each step of the subtitle pipeline"""
    
    print("🔍 DEBUGGING SUBTITLE PIPELINE")
    print("=" * 50)
    
    test_text = "This is a test subtitle. It should appear in the video. Let's see what happens."
    
    generator = VideoGenerator()
    
    print("Step 1: Testing script generation...")
    sentences = generator.split_article_into_sentences(test_text)
    print(f"✓ Sentences: {sentences}")
    
    print("\nStep 2: Testing audio generation...")
    try:
        audio_path, duration = generator.generate_audio_from_api(test_text)
        print(f"✓ Audio generated: {audio_path}, duration: {duration}s")
    except Exception as e:
        print(f"✗ Audio generation failed: {e}")
        return False
    
    print("\nStep 3: Testing SRT file generation...")
    try:
        # Create chunks for SRT generation (simulating the actual process)
        chunks = []
        words_per_chunk = len(test_text.split()) // 3  # Split into 3 chunks
        words = test_text.split()
        
        for i in range(0, len(words), max(1, words_per_chunk)):
            chunk_words = words[i:i + words_per_chunk]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "lines": [chunk_text],
                "start_time": i * (duration / 3),
                "end_time": (i + 1) * (duration / 3)
            })
        
        print(f"✓ Created {len(chunks)} subtitle chunks:")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: '{chunk['lines'][0]}' ({chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s)")
        
        # Generate SRT file
        srt_path = generator.generate_srt_file(sentences, duration)
        print(f"✓ SRT file created: {srt_path}")
        
        # Check SRT file content
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        print(f"✓ SRT content ({len(srt_content)} chars):")
        print("--- SRT Content ---")
        print(srt_content)
        print("--- End SRT Content ---")
        
    except Exception as e:
        print(f"✗ SRT generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nStep 4: Testing video generation with subtitles...")
    try:
        output_path = "/tmp/debug_subtitle_pipeline.mp4"
        
        # Test the full pipeline
        result = generator.generate_video_from_article(
            article=test_text,
            output_path=output_path,
            use_gpt_transcript=False,
            aspect_ratio="9:16"
        )
        
        print(f"✓ Video generated: {result}")
        
        # Verify the video exists and has reasonable size
        if os.path.exists(result):
            size = os.path.getsize(result)
            print(f"✓ Video file size: {size:,} bytes")
            
            # Check if subtitles are in the video using FFprobe
            print("\nStep 5: Verifying subtitles in generated video...")
            try:
                cmd = ['ffprobe', '-v', 'quiet', '-show_streams', '-select_streams', 's', result]
                probe_result = subprocess.run(cmd, capture_output=True, text=True)
                if probe_result.stdout:
                    print("✓ Subtitle streams found in video:")
                    print(probe_result.stdout)
                else:
                    print("⚠ No subtitle streams found in video metadata")
                    
                    # Check if subtitles are burned into the video (visual analysis)
                    print("Checking if subtitles are burned into video frames...")
                    # Extract a frame to check visually
                    frame_path = "/tmp/debug_frame.png"
                    frame_cmd = ['ffmpeg', '-y', '-i', result, '-vf', 'select=eq(n\\,30)', '-vframes', '1', frame_path]
                    subprocess.run(frame_cmd, capture_output=True)
                    if os.path.exists(frame_path):
                        print(f"✓ Frame extracted: {frame_path} (check manually for subtitles)")
                    else:
                        print("✗ Could not extract frame for visual inspection")
                        
            except Exception as e:
                print(f"✗ FFprobe failed: {e}")
        else:
            print("✗ Video file was not created")
            return False
            
    except Exception as e:
        print(f"✗ Video generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n🎯 DEBUGGING COMPLETE")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = debug_subtitle_pipeline()
    if success:
        print("✅ Pipeline debugging completed - check output above for issues")
    else:
        print("❌ Pipeline debugging failed - see errors above")