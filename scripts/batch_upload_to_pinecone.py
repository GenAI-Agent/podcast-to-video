#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量將 image_library 資料上傳到 Pinecone 的腳本
使用方法:
python batch_upload_to_pinecone.py --index-name "image-library" --namespace "production" --batch-size 50 --limit 100
"""

import argparse
import os
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.database.pinecone_handler import PineconeHandler

def main():
    parser = argparse.ArgumentParser(description='批量上傳 image_library 資料到 Pinecone')
    parser.add_argument('--index-name', required=True, help='Pinecone 索引名稱')
    parser.add_argument('--namespace', required=True, help='Pinecone 命名空間')
    parser.add_argument('--batch-size', type=int, default=50, help='每批處理的記錄數量 (預設: 50)')
    parser.add_argument('--limit', type=int, help='總共處理的記錄數量限制 (預設: 全部)')
    parser.add_argument('--offset', type=int, default=0, help='開始處理的偏移量 (預設: 0)')
    parser.add_argument('--auto-confirm', action='store_true', help='自動確認，不需要手動輸入 (預設: False)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("批量上傳 image_library 資料到 Pinecone")
    print("="*60)
    print(f"索引名稱: {args.index_name}")
    print(f"命名空間: {args.namespace}")
    print(f"批次大小: {args.batch_size}")
    print(f"記錄限制: {args.limit if args.limit else '無限制'}")
    print(f"偏移量: {args.offset}")
    print("="*60)
    
    # 確認是否繼續
    if not args.auto_confirm:
        confirm = input("確認要開始上傳嗎？ (y/N): ").lower().strip()
        if confirm != 'y':
            print("取消上傳操作")
            return
    else:
        print("自動確認模式，開始上傳...")
    
    # 初始化 PineconeHandler
    handler = PineconeHandler()
    
    # 執行批量上傳
    try:
        result = handler.batch_upload_image_library_to_pinecone(
            index_name=args.index_name,
            namespace=args.namespace, 
            batch_size=args.batch_size,
            limit=args.limit
        )
        
        print("\n" + "="*60)
        print("上傳完成！最終統計：")
        print("="*60)
        for key, value in result.items():
            print(f"{key}: {value}")
        print("="*60)
        
    except Exception as e:
        print(f"上傳過程中發生錯誤: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())