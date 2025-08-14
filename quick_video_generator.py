#!/usr/bin/env python3
"""
Quick Video Generator
A simple script to generate videos with different options
"""

import os
import sys
from datetime import datetime

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.generators.video_generator import VideoGenerator

def generate_video_quick(
    content_type="ai_demo",
    video_style="blur_horizontal", 
    overlay_text="Demo Video"
):
    """
    Quick video generation with predefined content
    
    Args:
        content_type: "ai_demo", "business", "tech", "custom"
        video_style: "blur_horizontal", "blur_vertical", "standard"
        overlay_text: Text to display on video
    """
    
    # Predefined content options
    content_options = {
        "ai_demo": """
        Artificial Intelligence is revolutionizing industries worldwide. 
        Machine learning algorithms process vast datasets to identify patterns and make predictions. 
        From healthcare diagnostics to autonomous vehicles, AI is creating unprecedented opportunities. 
        The future of technology lies in intelligent systems that augment human capabilities and solve complex problems.
        """,
        
        "business": """
        Digital transformation is reshaping the business landscape. 
        Companies embracing cloud technologies and data analytics gain competitive advantages. 
        Remote work and digital collaboration tools have become essential for modern organizations. 
        Success in today's market requires agility, innovation, and customer-centric approaches.
        """,
        
        "tech": """
        The technology sector continues to drive innovation across all industries. 
        Cloud computing, cybersecurity, and mobile applications are fundamental to modern business operations. 
        Emerging technologies like blockchain and quantum computing promise to revolutionize how we process information. 
        The intersection of hardware and software creates new possibilities for solving global challenges.
        """,
        
        "custom": "Enter your custom content here when prompted."
    }
    
    # Get content
    if content_type == "custom":
        article_content = input("Enter your article content: ")
    else:
        article_content = content_options.get(content_type, content_options["ai_demo"])
    
    # Video style settings
    if video_style == "blur_horizontal":
        use_blur = True
        aspect_ratio = "16:9"
        blur_strength = 24
    elif video_style == "blur_vertical":
        use_blur = True
        aspect_ratio = "9:16"
        blur_strength = 24
    else:  # standard
        use_blur = False
        aspect_ratio = "16:9"
        blur_strength = 0
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"generated_video_{content_type}_{timestamp}.mp4"
    
    print("🎬 Quick Video Generator")
    print("=" * 40)
    print(f"📝 Content Type: {content_type}")
    print(f"🎨 Video Style: {video_style}")
    print(f"📐 Aspect Ratio: {aspect_ratio}")
    print(f"🌫️  Blur Effect: {'Enabled' if use_blur else 'Disabled'}")
    print(f"📄 Output: {output_filename}")
    print("=" * 40)
    
    # Initialize generator
    generator = VideoGenerator()
    
    try:
        print("🚀 Starting video generation...")
        
        output_file = generator.generate_video_from_article(
            article=article_content,
            output_path=output_filename,
            use_gpt_transcript=True,  # Use AI to improve the transcript
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

def main():
    """Interactive main function"""
    print("🎬 Welcome to Quick Video Generator!")
    print("\nContent Options:")
    print("1. AI Demo (default)")
    print("2. Business")
    print("3. Technology")
    print("4. Custom")
    
    content_choice = input("\nSelect content type (1-4, default=1): ").strip()
    content_map = {"1": "ai_demo", "2": "business", "3": "tech", "4": "custom", "": "ai_demo"}
    content_type = content_map.get(content_choice, "ai_demo")
    
    print("\nVideo Style Options:")
    print("1. Blur Background - Horizontal (16:9)")
    print("2. Blur Background - Vertical (9:16)")  
    print("3. Standard (no blur)")
    
    style_choice = input("\nSelect video style (1-3, default=1): ").strip()
    style_map = {"1": "blur_horizontal", "2": "blur_vertical", "3": "standard", "": "blur_horizontal"}
    video_style = style_map.get(style_choice, "blur_horizontal")
    
    overlay_text = input("\nEnter overlay text (default='Demo Video'): ").strip()
    if not overlay_text:
        overlay_text = "Demo Video"
    
    # Generate the video
    result = generate_video_quick(content_type, video_style, overlay_text)
    
    if result:
        print(f"\n🎉 Success! Your video is ready: {result}")
    else:
        print("\n😞 Video generation failed. Check the error messages above.")

if __name__ == "__main__":
    main()