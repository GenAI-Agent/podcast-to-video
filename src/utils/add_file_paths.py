#!/usr/bin/env python3
"""
Script to add file_path field to JSON files by retrieving from ComfyUI history using task_id
"""

import json
import os
import requests
import time
from typing import Dict, Any, Optional

def get_file_info_from_comfy_history(task_id: str, max_attempts: int = 3) -> Optional[Dict[str, str]]:
    """
    Retrieve file name and path from ComfyUI history using task_id
    
    Args:
        task_id (str): The task ID to look up
        max_attempts (int): Maximum number of attempts
        
    Returns:
        Optional[Dict[str, str]]: Dictionary with 'file_name' and 'file_path', or None if not found
    """
    try:
        COMFY_URL = "https://6e0bf634b876.ngrok-free.app"
        history_url = f"{COMFY_URL}/history/{task_id}"
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(history_url)
                if response.status_code == 200:
                    history_data = response.json()
                    
                    # Check if the workflow is complete
                    prompt_data = history_data.get(task_id, {})
                    outputs = prompt_data.get('outputs', {})
                    
                    # Look for SaveImage node outputs (usually node "9" or similar)
                    for node_id, output in outputs.items():
                        if 'images' in output:
                            images = output['images']
                            if images and len(images) > 0:
                                image_info = images[0]
                                file_name = image_info.get('filename', '')
                                # Construct file path based on ComfyUI output structure
                                subfolder = image_info.get('subfolder', '')
                                if subfolder:
                                    file_path = f"C:\\Users\\x7048\\Documents\\ComfyUI\\output\\{subfolder}\\{file_name}"
                                else:
                                    file_path = f"C:\\Users\\x7048\\Documents\\ComfyUI\\output\\{file_name}"
                                
                                return {
                                    'file_name': os.path.splitext(file_name)[0],  # Remove extension
                                    'file_path': file_path
                                }
                
                time.sleep(1)  # Wait before retry
                
            except Exception as e:
                print(f"Error checking history for {task_id} (attempt {attempt + 1}): {str(e)}")
                time.sleep(1)
                
        return None
        
    except Exception as e:
        print(f"Error retrieving file info for {task_id}: {str(e)}")
        return None

def add_file_paths_to_json(json_file_path: str, output_file_path: str = None) -> None:
    """
    Add file_path field to all items in a JSON file by retrieving from ComfyUI history
    
    Args:
        json_file_path (str): Path to the input JSON file
        output_file_path (str): Path to save the updated JSON file (optional)
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If no output file specified, create one with "_updated" suffix
        if output_file_path is None:
            base_name = os.path.splitext(json_file_path)[0]
            output_file_path = f"{base_name}_updated.json"
        
        updated_items = 0
        
        # Process each category in the JSON
        for category, items in data.items():
            print(f"\n=== Processing category: {category} ===")
            print(f"Found {len(items)} items in this category")
            
            for i, item in enumerate(items):
                task_id = item.get('task_id', '')
                
                if task_id:
                    # Check if file_path already exists
                    if 'file_path' not in item:
                        print(f"  [{i+1}/{len(items)}] Retrieving file info for task {task_id}...")
                        file_info = get_file_info_from_comfy_history(task_id)
                        
                        if file_info:
                            # Update file_name if retrieved from ComfyUI
                            if file_info['file_name']:
                                item['file_name'] = file_info['file_name']
                            item['file_path'] = file_info['file_path']
                            updated_items += 1
                            print(f"    ✅ Added file_path: {file_info['file_path']}")
                        else:
                            print(f"    ❌ Failed to retrieve file info for task {task_id}")
                    else:
                        print(f"  [{i+1}/{len(items)}] file_path already exists for task {task_id}")
                else:
                    print(f"  [{i+1}/{len(items)}] No task_id found in item")
        
        # Save the updated data
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== Summary ===")
        print(f"Items updated with file_path: {updated_items}")
        print(f"Updated data saved to: {output_file_path}")
        
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def main():
    """Main function to process all JSON files in the root directory"""
    
    # List of JSON files to process with their output names
    json_files = [
        ("batch_image_data_20250804_111810_with_tags.json", "batch_image_data_20250804_111810_final.json"),
        ("batch_image_data_20250804_122941_with_tags.json", "batch_image_data_20250804_122941_final.json"), 
        ("batch_image_data_20250804_150029_with_tags.json", "batch_image_data_20250804_150029_final.json"),
        ("batch_image_data_20250804_155325_with_tags.json", "batch_image_data_20250804_155325_final.json"),
        ("batch_image_data_20250804_165413_with_tags.json", "batch_image_data_20250804_165413_final.json")
    ]
    
    print("Starting to add file_path fields to JSON files...")
    
    for input_file, output_file in json_files:
        if os.path.exists(input_file):
            print(f"\n{'='*60}")
            print(f"Processing: {input_file} -> {output_file}")
            print(f"{'='*60}")
            add_file_paths_to_json(input_file, output_file)
        else:
            print(f"Warning: File {input_file} not found, skipping...")
    
    print(f"\n{'='*60}")
    print("All files processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()