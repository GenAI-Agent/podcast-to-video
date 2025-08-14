#!/usr/bin/env python3
"""
Video Generation Demo Script
Generates a video with blur background effect using the VideoGenerator class
"""

import os
import sys
from datetime import datetime

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.generators.video_generator import VideoGenerator

def main():
    """Generate a demo video"""
    
    # Sample article content about technology and innovation
    sample_article = """
    Artificial Intelligence is revolutionizing how we work and live. 
    From autonomous vehicles to smart healthcare systems, AI technologies are creating unprecedented opportunities for innovation. 
    Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions with remarkable accuracy.
    
    The integration of AI in business operations is streamlining processes, reducing costs, and enhancing customer experiences. 
    Companies that embrace these technologies early are positioning themselves for competitive advantage in the digital economy.
    
    However, with great power comes great responsibility. As we advance AI capabilities, we must also address ethical considerations, 
    ensure data privacy, and work towards creating inclusive AI systems that benefit all of society.
    
    The future of AI holds immense promise, and we are just beginning to scratch the surface of what's possible.
    """
    
    # Initialize video generator
    print("🎬 Initializing Video Generator...")
    generator = VideoGenerator()
    
    try:
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"generated_video_{timestamp}.mp4"
        
        print(f"📝 Article content: {len(sample_article)} characters")
        print("🎵 Generating video with blur background effect...")
        
        # Generate video with blur background effect
        output_file = generator.generate_video_from_article(
            article=sample_article,
            output_path=output_filename,
            use_gpt_transcript=True,  # Use GPT to create a better transcript
            use_blur_background=True,  # Enable blur background effect
            overlay_text="AI Innovation Demo",  # Custom overlay text
            blur_strength=24,  # Blur strength
            aspect_ratio="16:9",  # Horizontal video format
            custom_srt_path=None,
            custom_audio_path=None
        )
        
        print(f"✅ Video generation completed successfully!")
        print(f"📁 Output file: {output_file}")
        print(f"📊 Video features:")
        print(f"   - Blur background effect")
        print(f"   - AI-generated transcript")
        print(f"   - Vector-searched images")
        print(f"   - Generated audio narration")
        print(f"   - Subtitle overlay")
        print(f"   - 16:9 aspect ratio")
        
        # Check if file was created successfully
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
            print(f"📏 File size: {file_size:.2f} MB")
        else:
            print("⚠️ Warning: Output file not found")
            
    except Exception as e:
        print(f"❌ Error during video generation: {str(e)}")
        print("💡 Make sure you have:")
        print("   - FFmpeg installed and accessible")
        print("   - Required environment variables set (.env file)")
        print("   - Pinecone API access configured")
        print("   - ElevenLabs API key (for audio generation)")
        raise
    finally:
        # Cleanup temporary files
        print("🧹 Cleaning up temporary files...")
        generator.cleanup()

if __name__ == "__main__":
    main()