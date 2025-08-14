#!/usr/bin/env python3
"""
Debug script to examine Pinecone metadata and understand the data structure
"""

import sys
import os

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.database.pinecone_handler import PineconeHandler

def debug_pinecone_search():
    """Debug Pinecone search to see what metadata is available"""
    print("🔍 Debugging Pinecone search results...")
    
    try:
        handler = PineconeHandler()
        
        # Test search with astrology-related terms
        test_queries = [
            "astrology",
            "constellation", 
            "zodiac",
            "Aries",
            "ram",
            "fire elements"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Searching for: '{query}'")
            results = handler.query_pinecone(
                query=query,
                metadata_filter={},
                index_name="image-library",
                namespace="description",
                top_k=3
            )
            
            if results:
                print(f"   Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"   {i}. Score: {result.get('score', 'N/A'):.3f}")
                    
                    metadata = result.get('metadata', {})
                    print(f"      Metadata keys: {list(metadata.keys())}")
                    
                    # Print key metadata fields
                    for key in ['file_path', 'task_id', 'id', 'uuid', 'image_id', 'category', 'description']:
                        if key in metadata:
                            value = metadata[key]
                            if isinstance(value, str) and len(value) > 100:
                                value = value[:100] + "..."
                            print(f"      {key}: {value}")
                    
                    print()
            else:
                print("   No results found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Pinecone debug: {e}")
        return False

def main():
    """Run debug analysis"""
    print("🔍 Pinecone Metadata Debug Analysis")
    print("=" * 50)
    
    success = debug_pinecone_search()
    
    if success:
        print("\n✅ Debug analysis completed")
    else:
        print("\n❌ Debug analysis failed")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
