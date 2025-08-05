#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to upload data from PostgreSQL to Pinecone
Will upload to two namespaces:
1. namespace='prompt' - using prompt as vector
2. namespace='description' - using description as vector
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.database.pinecone_handler import PineconeHandler

def main():
    # Initialize Pinecone handler
    handler = PineconeHandler()
    
    # Configuration
    index_name = "image-library"
    batch_size = 50
    
    print("=" * 80)
    print("Starting PostgreSQL to Pinecone upload process")
    print("=" * 80)
    print(f"Index: {index_name}")
    print(f"Batch size: {batch_size}")
    print("Will upload to namespaces: 'prompt' and 'description'")
    print("=" * 80)
    
    try:
        # Upload to both namespaces (prompt and description)
        # By not specifying namespace, it will upload to both
        result = handler.batch_upload_image_library_to_pinecone(
            index_name=index_name,
            namespace=None,  # None means upload to both 'prompt' and 'description'
            batch_size=batch_size,
            limit=None  # Process all records
        )
        
        print("\n" + "=" * 80)
        print("Upload process completed!")
        print(f"Final statistics: {result}")
        
    except Exception as e:
        print(f"\nError during upload process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()