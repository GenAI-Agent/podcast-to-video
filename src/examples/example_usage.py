#!/usr/bin/env python3
"""
簡化的使用範例
"""

import os
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.generators.video_generator import VideoGenerator

def run_example():
    # 設定你的 Pinecone 憑證
    PINECONE_API_KEY = "你的-pinecone-api-key"  # 請替換為你的 API key
    PINECONE_ENVIRONMENT = "你的-pinecone-environment"  # 請替換為你的環境
    
    # 你的文章內容
    article = input("請輸入你的文章內容: ")
    
    # 建立影片生成器
    generator = VideoGenerator(PINECONE_API_KEY, PINECONE_ENVIRONMENT)
    
    try:
        # 生成影片
        output_file = generator.generate_video_from_article(
            article, 
            "generated_video.mp4"
        )
        print(f"影片生成成功！儲存位置: {output_file}")
        
    except Exception as e:
        print(f"錯誤: {e}")
    finally:
        generator.cleanup()

if __name__ == "__main__":
    run_example()