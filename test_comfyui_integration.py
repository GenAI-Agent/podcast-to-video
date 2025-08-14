#!/usr/bin/env python3
"""
Test script to verify ComfyUI integration with RealtimeVideoGenerator
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append('/home/fluxmind/batch_image')

def test_realtime_generator():
    """Test the integrated RealtimeVideoGenerator"""
    from src.generators.Realtime_Video_Gen import RealtimeVideoGenerator
    
    print("=== Testing RealtimeVideoGenerator Integration ===\n")
    
    # Sample article
    sample_article = """
    Artificial Intelligence is transforming the way we create content.
    AI-powered tools can now generate images, write scripts, and even create videos.
    This technology is making content creation more accessible than ever before.
    """
    
    # Initialize with ComfyUI URL (update this with your actual URL)
    comfyui_url = "https://7fd6781ec07e.ngrok-free.app/api/prompt"
    print(f"ComfyUI URL: {comfyui_url}")
    
    try:
        generator = RealtimeVideoGenerator(comfyui_url=comfyui_url)
        print("✓ RealtimeVideoGenerator initialized successfully\n")
        
        # Test 1: Art style selection
        print("Test 1: Art Style Selection")
        result = generator.process_text_to_image_prompt(sample_article)
        print(f"Art Style: {result['art_style']}")
        print(f"Image Prompt: {result['image_prompt'][:100]}...\n")
        
        # Test 2: Scene breakdown
        print("Test 2: Scene Breakdown")
        scenes = generator.break_script_into_scenes(sample_article, 30)
        print(f"Generated {len(scenes)} scenes:")
        for i, scene in enumerate(scenes[:3]):  # Show first 3
            print(f"  Scene {i+1}: {scene['description'][:50]}... ({scene['duration']:.1f}s)")
        print()
        
        # Test 3: Full pipeline (without ComfyUI to test faster)
        print("Test 3: Full Pipeline (using database images)")
        video_path = generator.process_article_to_video(
            article=sample_article,
            output_path="test_ai_video.mp4",
            target_duration=20,  # Short test
            use_comfyui=False,   # Use database images for faster testing
            aspect_ratio="9:16"
        )
        print(f"✓ Video created: {video_path}")
        
        print("\n=== All Tests Passed! ===")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'generator' in locals():
            generator.cleanup()

def test_adapter():
    """Test the adapter"""
    from src.generators.realtime_adapter import RealtimeVideoGeneratorAdapter
    
    print("\n=== Testing RealtimeVideoGeneratorAdapter ===\n")
    
    try:
        adapter = RealtimeVideoGeneratorAdapter()
        print("✓ Adapter initialized successfully")
        
        # Test transcript generation
        sample_article = "AI is changing how we work and create content."
        transcript = adapter.generate_transcript_from_article(sample_article)
        print(f"✓ Transcript generated: {len(transcript)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Adapter test failed: {e}")
        return False
    
    finally:
        if 'adapter' in locals():
            adapter.cleanup()

def main():
    """Run all tests"""
    print("🚀 Testing ComfyUI Integration\n")
    
    # Check environment variables
    print("Environment Check:")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    comfyui_url = os.getenv("COMFYUI_URL")
    
    print(f"  Azure OpenAI Endpoint: {'✓' if azure_endpoint else '❌'}")
    print(f"  Azure OpenAI Key: {'✓' if azure_key else '❌'}")
    print(f"  ElevenLabs Key: {'✓' if elevenlabs_key else '❌'}")
    print(f"  ComfyUI URL: {comfyui_url or 'Not set (will use default)'}")
    print()
    
    # Run tests
    test1_passed = test_realtime_generator()
    test2_passed = test_adapter()
    
    if test1_passed and test2_passed:
        print("\n🎉 All integration tests passed!")
        print("\nYou can now use the ComfyUI integration by:")
        print("1. Setting backend_type='realtime' in the API")
        print("2. Setting use_comfyui=true for image generation")
        print("3. Or using the frontend ComfyUI checkbox")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()