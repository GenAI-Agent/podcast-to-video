#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to delete vectors from Pinecone based on sub_theme filter
Index: image-library
Namespace: lens-quant
Filter: sub_theme in [CRYPTO, WALLSTREET, BULLISH, BEAR_MARKET, RETAIL_SAVINGS]
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.database.pinecone_handler import PineconeHandler

def delete_vectors_by_sub_theme():
    """
    Delete vectors from Pinecone that match the specified sub_theme values
    """
    # Initialize Pinecone handler
    handler = PineconeHandler()
    
    # Configuration
    index_name = "image-library"
    namespace = "lens-quant"
    target_sub_themes = ["CRYPTO", "WALLSTREET", "BULLISH", "BEAR_MARKET", "RETAIL_SAVINGS"]
    
    print(f"Starting deletion process...")
    print(f"Index: {index_name}")
    print(f"Namespace: {namespace}")
    print(f"Target sub_themes: {target_sub_themes}")
    
    try:
        # Get Pinecone index
        index = handler._pc.Index(index_name, pool_threads=50)
        
        # Build metadata filter for OR condition
        metadata_filter = {
            "$or": [
                {"sub_theme": {"$eq": sub_theme}} for sub_theme in target_sub_themes
            ]
        }
        
        print(f"Metadata filter: {metadata_filter}")
        
        # First, query to see what we'll be deleting
        print("\nQuerying vectors to be deleted...")
        results = index.query(
            namespace=namespace,
            vector=[0] * 512,  # Dummy vector for metadata-only search
            top_k=10000,  # Large number to get all matches
            filter=metadata_filter,
            include_values=False,
            include_metadata=True
        )
        
        if not results["matches"]:
            print("No vectors found matching the filter criteria.")
            return
        
        print(f"Found {len(results['matches'])} vectors to delete:")
        
        # Show preview of what will be deleted
        for i, match in enumerate(results["matches"][:10]):  # Show first 10
            metadata = match.get("metadata", {})
            print(f"  {i+1}. ID: {match['id']}, sub_theme: {metadata.get('sub_theme', 'N/A')}, name: {metadata.get('name', 'N/A')}")
        
        if len(results["matches"]) > 10:
            print(f"  ... and {len(results['matches']) - 10} more")
        
        # Auto-confirm deletion (remove interactive prompt)
        print(f"\nProceeding with deletion of {len(results['matches'])} vectors...")
        
        # Extract IDs for deletion
        ids_to_delete = [match["id"] for match in results["matches"]]
        
        # Delete in batches (Pinecone has limits on batch operations)
        batch_size = 1000
        deleted_count = 0
        
        for i in range(0, len(ids_to_delete), batch_size):
            batch_ids = ids_to_delete[i:i + batch_size]
            print(f"Deleting batch {i//batch_size + 1}/{(len(ids_to_delete) + batch_size - 1)//batch_size} ({len(batch_ids)} vectors)...")
            
            try:
                index.delete(ids=batch_ids, namespace=namespace)
                deleted_count += len(batch_ids)
                print(f"Successfully deleted {len(batch_ids)} vectors")
            except Exception as e:
                print(f"Error deleting batch: {str(e)}")
                continue
        
        print(f"\nDeletion completed!")
        print(f"Total vectors deleted: {deleted_count}")
        
        # Verify deletion
        print("\nVerifying deletion...")
        verification_results = index.query(
            namespace=namespace,
            vector=[0] * 512,
            top_k=100,
            filter=metadata_filter,
            include_values=False,
            include_metadata=True
        )
        
        remaining_count = len(verification_results["matches"])
        if remaining_count == 0:
            print("✅ All targeted vectors have been successfully deleted!")
        else:
            print(f"⚠️  {remaining_count} vectors still remain (may need additional cleanup)")
            
    except Exception as e:
        print(f"Error during deletion process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    delete_vectors_by_sub_theme()