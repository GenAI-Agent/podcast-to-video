#!/usr/bin/env python3
"""
Demo script showcasing the new blur background video generation functionality.
This demonstrates the features copied from the Windows batch file:
- 9:16 vertical aspect ratio (mobile-friendly)
- Blur background effect with customizable strength
- Bold Arial font with black outline
- Text positioned 15% down from top
"""

import os
import sys
from typing import Optional

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.generators.video_generator import VideoGenerator


def create_blur_demo_video(
    article_text: str,
    output_filename: str = "blur_demo_video.mp4",
    overlay_text: str = "Blur Background Demo",
    blur_strength: int = 24,
    aspect_ratio: str = "9:16"
) -> Optional[str]:
    """
    Create a demo video with blur background effect
    
    Args:
        article_text: Text content for the video
        output_filename: Name of the output video file
        overlay_text: Text to overlay on the video
        blur_strength: Blur effect strength (higher = more blur)
        aspect_ratio: Video format ("9:16" for vertical, "16:9" for horizontal)
    
    Returns:
        Path to the generated video file, or None if failed
    """
    print("🎬 Starting Blur Background Video Demo")
    print("=" * 50)
    print(f"📐 Aspect Ratio: {aspect_ratio}")
    print(f"🌫️  Blur Strength: {blur_strength}")
    print(f"📝 Overlay Text: '{overlay_text}'")
    print(f"📄 Article Length: {len(article_text)} characters")
    print("=" * 50)
    
    # Initialize video generator
    generator = VideoGenerator()
    
    try:
        # Generate video with blur background effect
        output_path = generator.generate_video_from_article(
            article=article_text,
            output_path=output_filename,
            use_gpt_transcript=True,  # Use GPT to create better transcript
            use_blur_background=True,  # Enable blur background effect
            overlay_text=overlay_text,
            blur_strength=blur_strength,
            aspect_ratio=aspect_ratio
        )
        
        print("\n✅ Demo video created successfully!")
        print(f"📹 Output file: {output_path}")
        print(f"🎯 Features demonstrated:")
        print(f"   - Blur background effect (strength: {blur_strength})")
        print(f"   - {aspect_ratio} aspect ratio")
        print(f"   - Bold Arial font with black outline")
        print(f"   - Text positioned 15% from top")
        print(f"   - Enhanced subtitle styling")
        
        return output_path
        
    except Exception as e:
        print(f"\n❌ Error creating demo video: {str(e)}")
        return None
    finally:
        # Cleanup temporary files
        generator.cleanup()


def main():
    """Main demo function with sample content"""
    
    # Sample article content for demo
    demo_article = """
    Welcome to the future of video creation! This demo showcases an advanced blur background effect
    that transforms ordinary images into stunning visual experiences. 
    
    The blur background technique creates a cinematic depth of field effect, where the main subject
    remains sharp while the background is artistically blurred. This approach is perfect for 
    social media content, presentations, and mobile viewing.
    
    Key features include customizable blur strength, multiple aspect ratios, and professional
    typography with bold fonts and elegant outlines. The vertical 9:16 format is optimized 
    for Instagram Stories, TikTok, YouTube Shorts, and other mobile platforms.
    
    Experience the difference that professional video effects can make in your content creation workflow!
    """
    
    print("🎯 Blur Background Video Demo")
    print("This demo will create videos showcasing different blur effects and formats.\n")
    
    # Demo 1: Vertical format with strong blur (mobile-optimized)
    print("📱 Demo 1: Mobile-optimized vertical video")
    vertical_video = create_blur_demo_video(
        article_text=demo_article,
        output_filename="demo_vertical_blur.mp4",
        overlay_text="Mobile Ready",
        blur_strength=24,
        aspect_ratio="9:16"
    )
    
    if vertical_video:
        print(f"✅ Vertical demo created: {vertical_video}\n")
    else:
        print("❌ Failed to create vertical demo\n")
    
    # Demo 2: Horizontal format with medium blur (traditional format)
    print("🖥️  Demo 2: Traditional horizontal video")
    horizontal_video = create_blur_demo_video(
        article_text=demo_article,
        output_filename="demo_horizontal_blur.mp4",
        overlay_text="Desktop Ready",
        blur_strength=18,
        aspect_ratio="16:9"
    )
    
    if horizontal_video:
        print(f"✅ Horizontal demo created: {horizontal_video}\n")
    else:
        print("❌ Failed to create horizontal demo\n")
    
    # Demo 3: Vertical format with extreme blur for artistic effect
    print("🎨 Demo 3: Artistic high-blur vertical video")
    artistic_video = create_blur_demo_video(
        article_text=demo_article,
        output_filename="demo_artistic_blur.mp4",
        overlay_text="Artistic Vision",
        blur_strength=36,
        aspect_ratio="9:16"
    )
    
    if artistic_video:
        print(f"✅ Artistic demo created: {artistic_video}\n")
    else:
        print("❌ Failed to create artistic demo\n")
    
    print("🎉 Demo completed!")
    print("\nGenerated videos:")
    for filename in ["demo_vertical_blur.mp4", "demo_horizontal_blur.mp4", "demo_artistic_blur.mp4"]:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / (1024 * 1024)  # Size in MB
            print(f"  📹 {filename} ({file_size:.1f} MB)")
    
    print("\n📋 Technical details:")
    print("  - Blur effect: Gaussian blur with customizable strength")
    print("  - Font: Arial Bold with black outline (5px border)")
    print("  - Text position: 15% down from top, centered horizontally")
    print("  - Video quality: CRF 18 (high quality)")
    print("  - Frame rate: 30 FPS")
    print("  - Audio: AAC encoding")
    print("  - Subtitle styling: Enhanced for blur background visibility")


if __name__ == "__main__":
    main()