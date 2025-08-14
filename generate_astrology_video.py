#!/usr/bin/env python3
"""
Astrology Video Generator
Specialized script to generate astrology-themed videos using restricted image dataset
"""

import os
import sys
from datetime import datetime

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.generators.video_generator import VideoGenerator

def load_fixed_article():
    """Load the fixed article content about astrology"""
    article_path = "video-generator-app/src/fixed_article/article.txt"
    
    if os.path.exists(article_path):
        with open(article_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # Fallback astrology content if file not found
        return """
        Astrology is the ancient art of understanding how celestial bodies influence our lives and personalities. 
        Each zodiac sign carries unique characteristics and energies that shape who we are.
        
        From fiery Aries with its bold ram symbolism to mystical Pisces with flowing water elements,
        the twelve zodiac signs create a cosmic tapestry of human experience.
        
        The constellations above us tell stories of mythical creatures and divine archetypes.
        Taurus brings earthy stability, Gemini offers airy communication, and Cancer provides nurturing water energy.
        
        Leo shines with solar fire, Virgo grounds us with earth wisdom, and Libra seeks balance through air.
        Scorpio transforms through water's depths, Sagittarius aims high with fire's passion.
        
        Capricorn climbs mountains with earth's determination, Aquarius pours forth air's innovation,
        and Pisces swims in water's infinite compassion.
        
        These celestial patterns guide us through life's journey, offering insight into our deepest nature
        and our connection to the cosmic dance of the universe.
        """

def main():
    """Generate astrology video with restricted image dataset"""
    
    # Configuration
    json_dataset_file = "batch_image_data_20250808_162646.json"
    
    # Check if the JSON dataset exists
    if not os.path.exists(json_dataset_file):
        print(f"❌ Error: JSON dataset file not found: {json_dataset_file}")
        print("Please ensure the file is in the current directory.")
        return None
    
    # Load article content
    article_content = load_fixed_article()
    
    # Video settings optimized for astrology content
    use_blur = True
    aspect_ratio = "16:9"  # Horizontal format for better constellation viewing
    blur_strength = 20  # Slightly less blur to maintain mystical atmosphere
    overlay_text = "Astrology & Zodiac Signs"
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"astrology_video_{timestamp}.mp4"
    
    print("🌟 Astrology Video Generator")
    print("=" * 50)
    print(f"📝 Content: Astrology & Zodiac Signs")
    print(f"🎨 Video Style: Mystical Blur Background")
    print(f"📐 Aspect Ratio: {aspect_ratio}")
    print(f"🌫️  Blur Strength: {blur_strength}")
    print(f"🔒 Image Dataset: {json_dataset_file}")
    print(f"📄 Output: {output_filename}")
    print("=" * 50)
    
    # Initialize generator with restricted image dataset
    try:
        print("🚀 Initializing video generator with restricted image dataset...")
        generator = VideoGenerator(restricted_json_file=json_dataset_file)
        
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
            print(f"   ✓ Used restricted image dataset: {json_dataset_file}")
            print(f"   ✓ Generated astrology-themed video")
            print(f"   ✓ Applied mystical blur effects")
            print(f"   ✓ Output file: {output_file}")
        
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
        print("4. Verify the JSON dataset file exists and is valid")
        return None
        
    finally:
        # Cleanup
        if 'generator' in locals():
            generator.cleanup()

def interactive_mode():
    """Interactive mode for customizing video generation"""
    print("🌟 Astrology Video Generator - Interactive Mode")
    print("=" * 50)
    
    # Ask for customization options
    print("\nVideo Style Options:")
    print("1. Mystical Blur - Horizontal (16:9) - Recommended")
    print("2. Mystical Blur - Vertical (9:16)")  
    print("3. Standard (no blur)")
    
    style_choice = input("\nSelect video style (1-3, default=1): ").strip()
    
    if style_choice == "2":
        aspect_ratio = "9:16"
        use_blur = True
        blur_strength = 20
    elif style_choice == "3":
        aspect_ratio = "16:9"
        use_blur = False
        blur_strength = 0
    else:  # Default to option 1
        aspect_ratio = "16:9"
        use_blur = True
        blur_strength = 20
    
    overlay_text = input("\nEnter overlay text (default='Astrology & Zodiac Signs'): ").strip()
    if not overlay_text:
        overlay_text = "Astrology & Zodiac Signs"
    
    print(f"\n🎬 Generating astrology video with your settings...")
    print(f"   Style: {'Blur' if use_blur else 'Standard'}")
    print(f"   Aspect Ratio: {aspect_ratio}")
    print(f"   Overlay: {overlay_text}")
    
    # Generate with custom settings
    return main()

if __name__ == "__main__":
    # Check if user wants interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        result = interactive_mode()
    else:
        result = main()
    
    if result:
        print(f"\n🎉 Success! Your astrology video is ready: {result}")
        print("\n🌟 The video uses only images from the restricted astrology dataset!")
    else:
        print("\n😞 Video generation failed. Check the error messages above.")
        sys.exit(1)
