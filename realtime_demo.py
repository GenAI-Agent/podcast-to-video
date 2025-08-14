#!/usr/bin/env python3
"""
Real-time Image Generation Demo
Demonstrates the new real-time video generation features
"""

import os
import sys
from typing import Dict, List

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.generators.Realtime_Video_Gen import RealtimeVideoGenerator

def demo_realtime_generation():
    """Demonstrate the real-time image generation system"""
    
    print("🚀 Real-Time Video Generation Demo")
    print("=" * 50)
    
    # Initialize the generator
    generator = RealtimeVideoGenerator()
    
    # Sample user inputs
    user_inputs = [
        {
            "text": "A beautiful sunset over a calm ocean with gentle waves and seabirds flying overhead",
            "duration": 15,
            "description": "Nature scene"
        },
        {
            "text": "A cute cartoon cat playing with colorful yarn balls in a cozy living room",
            "duration": 10,
            "description": "Children's content"
        },
        {
            "text": "The process of photosynthesis in plants, showing sunlight, carbon dioxide, and oxygen",
            "duration": 20,
            "description": "Educational content"
        }
    ]
    
    try:
        for i, input_data in enumerate(user_inputs, 1):
            print(f"\n🎬 Demo {i}: {input_data['description']}")
            print("-" * 30)
            
            # Step 1: Process user input in real-time
            result = generator.process_user_input_realtime(
                user_input=input_data["text"],
                video_duration=input_data["duration"]
            )
            
            # Step 2: Display results
            print(f"\n📊 Results for Demo {i}:")
            print(f"   🎨 Art Style: {result['art_style']}")
            print(f"   📸 Total Images: {result['total_images']}")
            print(f"   ⏱️ Video Duration: {result['video_duration']}s")
            print(f"   📝 Prompts Generated: {len(result['comfyui_prompts'])}")
            
            # Step 3: Show timestamps and prompts
            print(f"\n🕐 Image Timestamps:")
            for j, (timestamp, prompt) in enumerate(zip(result['timestamps'], result['comfyui_prompts'])):
                print(f"   {j+1:2d}. [{timestamp:4.1f}s] {prompt[:60]}...")
            
            # Step 4: Get prompts ready for ComfyUI
            comfyui_prompts = generator.get_stored_prompts_for_comfyui()
            print(f"\n✅ {len(comfyui_prompts)} prompts ready for ComfyUI batch processing")
            
            # Reset for next demo (optional - shows art style selection behavior)
            if i < len(user_inputs):
                generator.reset_realtime_state()
                print("🔄 State reset for next demo")
        
        print(f"\n🎯 Demo Summary:")
        print(f"   - Prompt 1 (Art Style Selection): Runs once per content type")
        print(f"   - Prompt 2 (Image Generation): Runs for each image needed")
        print(f"   - Timestamps: Automatically calculated based on video duration")
        print(f"   - Storage: All prompts stored in list ready for ComfyUI")
        
        print(f"\n💡 Usage Tips:")
        print(f"   - Art style is cached until reset_realtime_state() is called")
        print(f"   - Use get_stored_prompts_for_comfyui() to get all prompts")
        print(f"   - Use get_image_timestamps() to get timing information")
        print(f"   - Use batch_generate_images_with_comfyui() for actual generation")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        generator.cleanup()

def show_api_usage():
    """Show the key API methods for real-time generation"""
    
    print(f"\n🔧 Key API Methods:")
    print("=" * 50)
    
    api_methods = [
        {
            "method": "process_user_input_realtime(user_input, video_duration)",
            "description": "Main method - processes user input and generates all data",
            "returns": "Dict with art_style, image_data, comfyui_prompts, timestamps"
        },
        {
            "method": "generate_realtime_images(user_input, duration, images_per_second)",
            "description": "Generates image prompts with timestamps",
            "returns": "List of image data dictionaries"
        },
        {
            "method": "get_stored_prompts_for_comfyui()",
            "description": "Get all stored prompts ready for ComfyUI",
            "returns": "List[str] of image prompts"
        },
        {
            "method": "get_image_timestamps()",
            "description": "Get timing information for video synchronization",
            "returns": "List[float] of timestamps in seconds"
        },
        {
            "method": "batch_generate_images_with_comfyui(image_data)",
            "description": "Send all prompts to ComfyUI for generation",
            "returns": "List[str] of generated image file paths"
        },
        {
            "method": "reset_realtime_state()",
            "description": "Reset art style and prompts for new session",
            "returns": "None"
        }
    ]
    
    for method in api_methods:
        print(f"\n📋 {method['method']}")
        print(f"   Description: {method['description']}")
        print(f"   Returns: {method['returns']}")

if __name__ == "__main__":
    print("🎥 Real-Time Video Generation System")
    print("Demonstrates Prompt 1 (once) + Prompt 2 (multiple) + Timestamps")
    print("=" * 70)
    
    # Run the demo
    demo_realtime_generation()
    
    # Show API usage
    show_api_usage()
    
    print(f"\n✅ Demo completed! Check the output above to see how the system works.")