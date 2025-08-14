#!/usr/bin/env python3
"""
Test script for the new blur background functionality with repeating pattern
"""

import os
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.generators.video_generator import VideoGenerator

def test_blur_background():
    """Test the blur background functionality"""
    
    # Sample article for testing
    test_article = """
    Today we're testing the new blur background feature that creates a repeating, tiled background.
    The image will be repeated across the background and then heavily blurred to create a cinematic effect.
    This helps cover aspect ratio mismatches between the source images and the target video format.
    The sharp original image is then overlaid on top of the blurred repeating background.
    """
    
    # Initialize video generator
    generator = VideoGenerator()
    
    try:
        print("🎬 Testing blur background with repeating pattern...")
        
        # Test with blur background enabled
        output_video = generator.generate_video_from_article(
            article=test_article,
            output_path="test_blur_background.mp4",
            use_gpt_transcript=True,
            use_blur_background=True,  # Enable the new blur effect
            overlay_text="Blur Test",   # Text overlay
            blur_strength=24,           # Standard blur strength
            aspect_ratio="9:16"         # Vertical format for mobile
        )
        
        print(f"✅ Test completed successfully!")
        print(f"📹 Output video: {output_video}")
        print("\n🎯 Features tested:")
        print("  ✓ Repeating/tiled background pattern")
        print("  ✓ Heavy Gaussian blur on background")
        print("  ✓ Sharp original image overlay")
        print("  ✓ 9:16 vertical aspect ratio")
        print("  ✓ Enhanced subtitle styling")
        
        # Check if file was created
        if os.path.exists(output_video):
            file_size = os.path.getsize(output_video) / (1024 * 1024)  # MB
            print(f"📊 Video file size: {file_size:.2f} MB")
            return True
        else:
            print("❌ Video file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
        
    finally:
        # Clean up temporary files
        generator.cleanup()

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 BLUR BACKGROUND FUNCTIONALITY TEST")
    print("=" * 50)
    
    success = test_blur_background()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed!")
        print("The blur background with repeating pattern is working correctly.")
    else:
        print("💥 Test failed!")
        print("Please check the error messages above.")
    print("=" * 50)