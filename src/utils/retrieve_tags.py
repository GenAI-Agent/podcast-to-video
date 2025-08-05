#!/usr/bin/env python3
"""
Script to retrieve tags from ComfyUI for all task IDs in a JSON file
"""

import json
import time
import requests
from typing import Optional, Dict, List
import os

def get_tags_from_comfy_history(prompt_id: str, max_attempts: int = 10) -> Optional[str]:
    """
    Retrieve WD14 Tagger tags from ComfyUI history after image generation
    
    Args:
        prompt_id (str): The prompt ID returned from ComfyUI
        max_attempts (int): Maximum number of attempts to check for completion
        
    Returns:
        Optional[str]: Generated tags string, or None if not found
    """
    try:
        # Extract base URL from COMFY_URL
        COMFY_URL = "https://6e0bf634b876.ngrok-free.app/api/prompt"
        base_url = COMFY_URL.replace("/api/prompt", "")
        history_url = f"{base_url}/history/{prompt_id}"
        
        for attempt in range(max_attempts):
            try:
                history_response = requests.get(history_url)
                if history_response.status_code == 200:
                    history_data = history_response.json()
                    
                    # Check if the workflow is complete
                    prompt_data = history_data.get(prompt_id, {})
                    outputs = prompt_data.get('outputs', {})
                    
                    # Look for WD14 Tagger output (node "39" in our workflow)
                    if '39' in outputs:
                        tagger_output = outputs['39']
                        if 'tags' in tagger_output:
                            tags = tagger_output['tags'][0] if isinstance(tagger_output['tags'], list) else tagger_output['tags']
                            return tags
                
                # Wait before next attempt
                time.sleep(2)
                
            except Exception as e:
                print(f"Error checking history (attempt {attempt + 1}): {str(e)}")
                time.sleep(2)
                
        print(f"Could not retrieve tags for prompt_id {prompt_id} after {max_attempts} attempts")
        return None
        
    except Exception as e:
        print(f"Error retrieving tags from ComfyUI history: {str(e)}")
        return None

def retrieve_tags_for_json_file(json_file_path: str, output_file_path: str = None, max_items: int = None) -> None:
    """
    Retrieve tags for all task IDs in a JSON file and update the file
    
    Args:
        json_file_path (str): Path to the JSON file containing task IDs
        output_file_path (str): Path to save the updated JSON file (optional)
        max_items (int): Maximum number of items to process (optional)
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If no output file specified, create one with "_with_tags" suffix
        if output_file_path is None:
            base_name = os.path.splitext(json_file_path)[0]
            output_file_path = f"{base_name}_with_tags.json"
        
        total_items = 0
        updated_items = 0
        
        # Process each category in the JSON
        for category, items in data.items():
            print(f"\n=== Processing category: {category} ===")
            print(f"Found {len(items)} items in this category")
            
            # Limit items if max_items is specified
            if max_items is not None:
                items = items[:max_items]
                print(f"Limiting to first {max_items} items")
            
            for i, item in enumerate(items):
                total_items += 1
                task_id = item.get('task_id')
                current_tags = item.get('tags', '')
                
                # Skip if tags are already retrieved (not "No tags retrieved")
                if current_tags and current_tags != "No tags retrieved":
                    print(f"  [{i+1}/{len(items)}] Task {task_id}: Tags already exist, skipping")
                    continue
                
                if task_id:
                    print(f"  [{i+1}/{len(items)}] Retrieving tags for task {task_id}...")
                    tags = get_tags_from_comfy_history(task_id)
                    
                    if tags:
                        item['tags'] = tags
                        updated_items += 1
                        print(f"    ✅ Retrieved tags: {tags[:100]}..." if len(tags) > 100 else f"    ✅ Retrieved tags: {tags}")
                    else:
                        print(f"    ❌ Failed to retrieve tags for task {task_id}")
                else:
                    print(f"  [{i+1}/{len(items)}] No task_id found in item")
        
        # Save the updated data
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== Summary ===")
        print(f"Total items processed: {total_items}")
        print(f"Items updated with tags: {updated_items}")
        print(f"Updated data saved to: {output_file_path}")
        
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def main():
    """Main function to run the tag retrieval script"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python retrieve_tags.py <json_file_path> [output_file_path] [max_items]")
        print("Example: python retrieve_tags.py batch_image_data_20250804_111810.json")
        print("Example: python retrieve_tags.py batch_image_data_20250804_111810.json output.json 43")
        return
    
    json_file_path = sys.argv[1]
    output_file_path = sys.argv[2] if len(sys.argv) > 2 else None
    max_items = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    print(f"Starting tag retrieval for: {json_file_path}")
    if max_items:
        print(f"Limiting to {max_items} items")
    retrieve_tags_for_json_file(json_file_path, output_file_path, max_items)

if __name__ == "__main__":
    main() 