#!/usr/bin/env python3
"""
Simple example showing how to use the new blur background video generation feature.
This replicates the functionality from the Windows batch file in Python.
"""

import os
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.generators.video_generator import VideoGenerator


def main():
    """Simple example of blur background video generation"""
    
    # Sample content
    article = """
    This is a demonstration of the new blur background video effect. 
    The background creates a beautiful cinematic depth of field while keeping 
    the main content sharp and readable. Perfect for social media and mobile viewing!
    """
    
    # Initialize video generator
    generator = VideoGenerator()
    
    try:
        print("🎬 Creating video with blur background effect...")
        
        # Generate video with blur background
        # This replicates the batch file functionality:
        # - 9:16 aspect ratio (1080x1920)
        # - Gaussian blur strength 24
        # - Arial Bold font with black outline
        # - Text positioned 15% from top
        output_video = generator.generate_video_from_article(
            article=article,
            output_path="blur_example_output.mp4",
            use_gpt_transcript=True,
            use_blur_background=True,  # Enable the blur effect
            overlay_text="Blur Demo",  # Text overlay (like batch file)
            blur_strength=24,          # Same blur strength as batch file
            aspect_ratio="9:16"        # Vertical format like batch file
        )
        
        print(f"✅ Video created successfully: {output_video}")
        print("\n🎯 Features applied:")
        print("  ✓ Blur background effect (strength: 24)")
        print("  ✓ 9:16 vertical aspect ratio (1080x1920)")
        print("  ✓ Arial Bold font with black outline")
        print("  ✓ Text positioned 15% from top")
        print("  ✓ Enhanced subtitle styling")
        print("  ✓ Perfect for mobile/social media")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        # Clean up temporary files
        generator.cleanup()


if __name__ == "__main__":
    main()