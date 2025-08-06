#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to import JSON files from root directory to PostgreSQL database
"""

import json
import os
import glob
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.database.postgres_handler import batch_insert_to_postgres

def process_json_files_from_root():
    """
    Process all batch_image_data_*_final.json files in the root directory
    and import them to PostgreSQL
    """
    # Find all JSON files in root directory matching the pattern
    json_files = glob.glob(str(project_root / "*_final.json"))
    
    if not json_files:
        print("No JSON files found in root directory")
        return
    
    print("Found JSON files:")
    for file in sorted(json_files):
        print(f"  - {os.path.basename(file)}")
    
    print(f"\nTotal files: {len(json_files)}")
    print("\n" + "="*80)
    print("Starting PostgreSQL import process...")
    print("="*80)
    
    all_batch_data = []
    
    for json_file in sorted(json_files):
        print(f"\nProcessing: {os.path.basename(json_file)}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Prepare batch data for this file
            batch_data = []
            
            for theme, items in data.items():
                for item in items:
                    task_id = item.get('task_id')
                    file_name = item.get('file_name')
                    file_path = item.get('file_path')
                    prompt = item.get('prompt', '')
                    tags = item.get('tags', '')
                    
                    if not task_id:
                        print(f"Warning: Missing task_id in item: {item}")
                        continue
                    
                    # Prepare data record
                    record = {
                        'id': task_id,  # Use task_id as id
                        'name': file_name or f"{theme}_{task_id[:8]}",  # Use file_name as name
                        'file_path': file_path or '',  # Use JSON file_path
                        'prompt': prompt,
                        'description': tags,  # Use tags as description
                        'theme': 'lens_quant',  # All records have theme 'lens_quant'
                        'sub_theme': theme,  # JSON key as sub_theme (e.g., BEAR_MARKET, CRYPTO)
                        'status': 'active'
                    }
                    
                    batch_data.append(record)
            
            all_batch_data.extend(batch_data)
            print(f"  ✅ Prepared {len(batch_data)} records from this file")
            
            # Show sample records from first file
            if json_file == json_files[0] and batch_data:
                print("\n  Sample records:")
                for i, record in enumerate(batch_data[:3]):
                    print(f"    Record {i+1}:")
                    print(f"      - ID: {record['id']}")
                    print(f"      - Name: {record['name']}")
                    print(f"      - Sub-theme: {record['sub_theme']}")
                    print(f"      - File path: {record['file_path'][:80]}..." if record['file_path'] else "      - File path: (empty)")
                
        except Exception as e:
            print(f"  ❌ Error processing {json_file}: {e}")
    
    if all_batch_data:
        print(f"\n{'='*80}")
        print(f"Total records to insert: {len(all_batch_data)}")
        
        # Show theme distribution
        theme_count = {}
        for record in all_batch_data:
            sub_theme = record['sub_theme']
            theme_count[sub_theme] = theme_count.get(sub_theme, 0) + 1
        
        print("\nRecords per sub-theme:")
        for theme, count in sorted(theme_count.items()):
            print(f"  - {theme}: {count} records")
        
        # Confirm before proceeding
        print(f"\n{'='*80}")
        print("Starting PostgreSQL import...")
        
        try:
            batch_insert_to_postgres(all_batch_data)
            print("\n✅ Import completed successfully!")
            print(f"   Imported {len(all_batch_data)} records to PostgreSQL")
        except Exception as e:
            print(f"\n❌ Import failed: {e}")
            return
            
    else:
        print("No data to import")

if __name__ == "__main__":
    process_json_files_from_root()