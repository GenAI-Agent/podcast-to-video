#!/usr/bin/env python3
"""
Practical Astrology Video Generator
Uses the best available astrology-related images from the existing Pinecone database
"""

import os
import sys
from datetime import datetime

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.generators.video_generator import VideoGenerator

def create_astrology_content():
    """Create focused astrology content that matches available images"""
    return """
    Astrology reveals the cosmic connections that shape our lives through the ancient wisdom of the stars.
    
    The zodiac signs represent twelve distinct personality archetypes, each with unique characteristics and elemental associations.
    Fire signs like Aries, Leo, and Sagittarius bring passion and energy to the cosmic wheel.
    Earth signs including Taurus, Virgo, and Capricorn provide stability and grounding.
    Air signs such as Gemini, Libra, and Aquarius offer communication and intellectual insight.
    Water signs like Cancer, Scorpio, and Pisces flow with emotion and intuition.
    
    Each constellation tells an ancient story, connecting us to mythological traditions that span cultures and centuries.
    The ram of Aries charges forward with pioneering spirit, while the bull of Taurus stands firm in earthly wisdom.
    Gemini's twins dance in duality, and Cancer's crab protects with nurturing care.
    
    Leo's lion roars with solar confidence, Virgo's maiden brings analytical precision.
    Libra's scales seek perfect balance, while Scorpio's scorpion transforms through intensity.
    Sagittarius aims arrows of truth toward distant horizons, and Capricorn's goat climbs mountains of ambition.
    
    Aquarius pours forth innovative ideas like water from an eternal vessel.
    Pisces swims in the depths of imagination and spiritual connection.
    
    Through understanding these celestial patterns, we gain insight into our deepest nature
    and our place in the grand cosmic design that connects all living beings.
    """

def main():
    """Generate practical astrology video using available images"""
    
    # Configuration for practical approach
    json_dataset_file = "batch_image_data_20250808_162646.json"
    
    # Load article content
    article_content = create_astrology_content()
    
    # Video settings optimized for available content
    use_blur = True
    aspect_ratio = "16:9"
    blur_strength = 18  # Moderate blur for mystical effect
    overlay_text = "Astrology & Cosmic Wisdom"
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"astrology_practical_{timestamp}.mp4"
    
    print("🌟 Practical Astrology Video Generator")
    print("=" * 55)
    print(f"📝 Content: Astrology & Cosmic Wisdom")
    print(f"🎨 Video Style: Mystical with Available Images")
    print(f"📐 Aspect Ratio: {aspect_ratio}")
    print(f"🌫️  Blur Strength: {blur_strength}")
    print(f"🔍 Image Source: Best Available from Pinecone")
    print(f"📄 Output: {output_filename}")
    print("=" * 55)
    
    # Initialize generator with fallback capability
    try:
        print("🚀 Initializing video generator...")
        
        # Try with restricted mode first, but with fallback enabled
        if os.path.exists(json_dataset_file):
            print(f"🔒 Attempting restricted mode with {json_dataset_file}")
            generator = VideoGenerator(restricted_json_file=json_dataset_file)
        else:
            print("📂 Using standard mode (no JSON restrictions)")
            generator = VideoGenerator()
        
        print("🎬 Starting astrology video generation...")
        
        output_file = generator.generate_video_from_article(
            article=article_content,
            output_path=output_filename,
            use_gpt_transcript=True,  # Use AI to improve the transcript
            use_blur_background=use_blur,
            overlay_text=overlay_text,
            blur_strength=blur_strength,
            aspect_ratio=aspect_ratio
        )
        
        print("✅ Astrology video generation completed!")
        print(f"📁 Saved as: {output_file}")
        
        # Check file size and provide summary
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"📊 File size: {file_size:.2f} MB")
            
            # Print success summary
            print("\n🎉 Success Summary:")
            print(f"   ✓ Generated astrology-themed video")
            print(f"   ✓ Used best available cosmic/mystical images")
            print(f"   ✓ Applied mystical blur effects")
            print(f"   ✓ Output file: {output_file}")
            
            # Provide viewing instructions
            print("\n📺 To view your video:")
            print(f"   • File location: {os.path.abspath(output_file)}")
            print("   • The video combines astrology content with cosmic imagery")
            print("   • Mystical blur effects enhance the spiritual atmosphere")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error generating astrology video: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure FFmpeg is installed and in your PATH")
        print("2. Check your .env file for required API keys:")
        print("   - AZURE_OPENAI_ENDPOINT")
        print("   - AZURE_OPENAI_API_KEY") 
        print("   - ELEVENLABS_API_KEY")
        print("   - PINECONE_API_KEY")
        print("3. Ensure you have internet connection for API calls")
        print("4. Check that Pinecone database contains some images")
        
        # Print additional debug info
        print(f"\n🔍 Debug info:")
        print(f"   - JSON file exists: {os.path.exists(json_dataset_file)}")
        print(f"   - Current directory: {os.getcwd()}")
        
        return None
        
    finally:
        # Cleanup
        if 'generator' in locals():
            generator.cleanup()

def quick_test():
    """Quick test to verify the setup works"""
    print("🧪 Running quick setup test...")
    
    try:
        from src.database.pinecone_handler import PineconeHandler
        handler = PineconeHandler()
        
        # Test a simple search
        results = handler.query_pinecone(
            query="mystical cosmic",
            metadata_filter={},
            index_name="image-library", 
            namespace="description",
            top_k=1
        )
        
        if results:
            print(f"✅ Pinecone connection working - found {len(results)} images")
            return True
        else:
            print("⚠️ Pinecone connected but no images found")
            return False
            
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        return False

if __name__ == "__main__":
    # Run quick test first
    if not quick_test():
        print("\n❌ Setup test failed. Please check your configuration.")
        sys.exit(1)
    
    print("\n" + "="*55)
    
    # Generate the video
    result = main()
    
    if result:
        print(f"\n🎉 Success! Your astrology video is ready: {result}")
        print("\n🌟 This video uses the best available cosmic and mystical images!")
        print("🎬 The content focuses on astrology and zodiac wisdom.")
    else:
        print("\n😞 Video generation failed. Check the error messages above.")
        sys.exit(1)
