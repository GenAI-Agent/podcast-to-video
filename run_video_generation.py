#!/usr/bin/env python3
"""
Automated video generation script - runs without user interaction
"""

import os
import sys
from datetime import datetime

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.generators.video_generator import VideoGenerator

def main():
    # Configuration - using default AI demo content
    article_content = """
    Artificial Intelligence is revolutionizing industries worldwide. 
    Machine learning algorithms process vast datasets to identify patterns and make predictions. 
    From healthcare diagnostics to autonomous vehicles, AI is creating unprecedented opportunities. 
    The future of technology lies in intelligent systems that augment human capabilities and solve complex problems.
    """
    
    # Video settings
    use_blur = True
    aspect_ratio = "16:9"
    blur_strength = 24
    overlay_text = "AI Technology Demo"
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"generated_video_ai_demo_{timestamp}.mp4"
    
    print("🎬 Automated Video Generator")
    print("=" * 40)
    print(f"📝 Content Type: AI Demo")
    print(f"🎨 Video Style: Blur Horizontal")
    print(f"📐 Aspect Ratio: {aspect_ratio}")
    print(f"🌫️  Blur Effect: Enabled")
    print(f"📄 Output: {output_filename}")
    print("=" * 40)
    
    # Initialize generator
    generator = VideoGenerator()
    
    try:
        print("🚀 Starting video generation...")
        
        output_file = generator.generate_video_from_article(
            article=article_content,
            output_path=output_filename,
            use_gpt_transcript=True,
            use_blur_background=use_blur,
            overlay_text=overlay_text,
            blur_strength=blur_strength,
            aspect_ratio=aspect_ratio
        )
        
        print("✅ Video generation completed!")
        print(f"📁 Saved as: {output_file}")
        
        # Check file size
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"📊 File size: {file_size:.2f} MB")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure FFmpeg is installed and in your PATH")
        print("2. Check your .env file for API keys:")
        print("   - AZURE_OPENAI_ENDPOINT")
        print("   - AZURE_OPENAI_API_KEY") 
        print("   - ELEVENLABS_API_KEY")
        print("   - PINECONE_API_KEY")
        print("3. Ensure you have internet connection for API calls")
        return None
        
    finally:
        # Cleanup
        generator.cleanup()

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n🎉 Success! Your video is ready: {result}")
    else:
        print("\n😞 Video generation failed. Check the error messages above.")