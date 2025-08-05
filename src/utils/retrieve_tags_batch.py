import json
import requests
import time
from typing import Optional, List
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
                if attempt < max_attempts - 1:  # Don't wait on the last attempt
                    time.sleep(2)
                
            except Exception as e:
                print(f"Error checking history for {prompt_id} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(2)
                
        print(f"Could not retrieve tags for prompt_id {prompt_id} after {max_attempts} attempts")
        return None
        
    except Exception as e:
        print(f"Error retrieving tags from ComfyUI history: {str(e)}")
        return None

def retrieve_tags_for_file(json_file_path: str, output_file_path: str = None):
    """
    Retrieve tags for all task IDs in a single JSON file and update the file
    
    Args:
        json_file_path (str): Path to the input JSON file
        output_file_path (str): Path to save the updated JSON file (defaults to same as input)
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing file: {json_file_path}")
        print(f"{'='*60}")
        
        # Check if file exists
        if not os.path.exists(json_file_path):
            print(f"❌ File not found: {json_file_path}")
            return
        
        # Load the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if output_file_path is None:
            output_file_path = json_file_path.replace('.json', '_with_tags.json')
        
        total_tasks = 0
        successful_retrievals = 0
        
        # Process each category in the JSON
        for category_key, items in data.items():
            print(f"\n=== Processing {category_key} category ===")
            total_tasks += len(items)
            
            for i, item in enumerate(items):
                task_id = item.get('task_id')
                current_tags = item.get('tags', 'No tags retrieved')
                
                print(f"Processing item {i+1}/{len(items)}: {task_id}")
                
                # Only retrieve tags if they haven't been retrieved yet
                if current_tags == "No tags retrieved" or not current_tags:
                    print(f"Retrieving tags for task_id: {task_id}")
                    retrieved_tags = get_tags_from_comfy_history(task_id)
                    
                    if retrieved_tags:
                        item['tags'] = retrieved_tags
                        successful_retrievals += 1
                        print(f"✅ Successfully retrieved tags: {retrieved_tags[:100]}..." if len(retrieved_tags) > 100 else f"✅ Successfully retrieved tags: {retrieved_tags}")
                    else:
                        print(f"❌ Could not retrieve tags for {task_id}")
                else:
                    print(f"⏭️  Tags already exist: {current_tags[:50]}..." if len(current_tags) > 50 else f"⏭️  Tags already exist: {current_tags}")
                    successful_retrievals += 1
        
        # Save the updated JSON file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 File processing completed!")
        print(f"📊 Results for {json_file_path}: {successful_retrievals}/{total_tasks} tasks have tags")
        print(f"💾 Updated file saved as: {output_file_path}")
        
        return successful_retrievals, total_tasks
        
    except Exception as e:
        print(f"Error processing JSON file {json_file_path}: {str(e)}")
        return 0, 0

def retrieve_all_tags_batch(json_files: List[str]):
    """
    Retrieve tags for multiple JSON files
    
    Args:
        json_files (List[str]): List of JSON file paths to process
    """
    total_successful = 0
    total_tasks = 0
    
    print("🚀 Starting batch tag retrieval process...")
    print(f"📁 Processing {len(json_files)} files")
    
    for json_file in json_files:
        successful, tasks = retrieve_tags_for_file(json_file)
        total_successful += successful
        total_tasks += tasks
        
        # Add a small delay between files to be respectful to the API
        time.sleep(1)
    
    print(f"\n{'='*80}")
    print(f"🏁 BATCH PROCESSING COMPLETE!")
    print(f"📊 Overall Results: {total_successful}/{total_tasks} total tasks have tags")
    print(f"📈 Success Rate: {(total_successful/total_tasks*100):.1f}%" if total_tasks > 0 else "No tasks processed")
    print(f"{'='*80}")

if __name__ == "__main__":
    # List of JSON files to process
    json_files = [
        "batch_image_data_20250801_133312.json",  # RETAIL_SAVINGS
        "batch_image_data_20250801_155450.json",  # STRATEGIES
        "batch_image_data_20250801_170613.json",  # RISK_VS_REWARD
    ]
    
    # Remove duplicates while preserving order
    unique_files = []
    for file in json_files:
        if file not in unique_files:
            unique_files.append(file)
    
    retrieve_all_tags_batch(unique_files) 