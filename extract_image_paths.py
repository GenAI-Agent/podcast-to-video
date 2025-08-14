#!/usr/bin/env python3
"""
Extract image paths from task collection results
"""

from task_collector import collect_task_results
import json

def extract_image_paths_from_results(results_dict):
    """
    Extract image_paths list from task collection results
    
    Args:
        results_dict (dict): Results from wait_and_collect_task_results
    
    Returns:
        list: List of file paths from completed tasks
    """
    image_paths = []
    
    if 'completed_tasks' in results_dict:
        for task in results_dict['completed_tasks']:
            if task.get('file_path'):
                image_paths.append(task['file_path'])
    
    return image_paths

def extract_image_paths_from_json_file(json_filename):
    """
    Extract image_paths from a JSON file containing task results
    
    Args:
        json_filename (str): Path to JSON file with task results
    
    Returns:
        list: List of file paths from completed tasks
    """
    try:
        with open(json_filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return extract_image_paths_from_results(results)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return []

def collect_and_extract_image_paths(task_ids_list, output_filename="collected_results.json"):
    """
    Collect task results and directly return image_paths list
    
    Args:
        task_ids_list: List of task IDs
        output_filename: JSON output filename
    
    Returns:
        list: List of image file paths
    """
    print(f"🔄 Collecting results for {len(task_ids_list)} tasks...")
    
    # Collect results using existing function
    results = collect_task_results(task_ids_list, output_filename)
    
    # Extract image paths
    image_paths = extract_image_paths_from_results(results)
    
    print(f"✅ Extracted {len(image_paths)} image paths")
    print(f"📄 Results saved to: {output_filename}")
    
    return image_paths

if __name__ == "__main__":
    # Example usage with your task IDs
    task_ids = [
        '1e52c41f-16fb-45b5-ad22-79135434196e', 
        'ea47d7d9-3fd0-427b-a982-50d2e9c1a5f7', 
        '56668548-325b-4178-b40f-ebdd99157fda', 
        '7d9b35c6-bc25-4efc-93be-1f3bf1f90b01', 
        '6d98132b-59f8-4030-b000-916be9dfcca3',
        '214e9e82-f16c-45d9-8236-70a050ec2c16', 
        '98115177-3222-47f1-8b88-6587faa6620e', 
        '6b98d57f-c713-4558-b9bb-aa1812ad646b', 
        'd783f431-fd23-43f8-8ee0-50965bc1d6d3', 
        '51550706-c373-4c28-ad7f-0567cb32043a'
    ]
    
    # Method 1: Collect and extract in one step
    image_paths = collect_and_extract_image_paths(task_ids, "your_task_results.json")
    print("\n🎯 Image Paths:")
    print(f"image_paths = {image_paths}")
    
    # Method 2: Extract from existing JSON file
    # image_paths = extract_image_paths_from_json_file("your_task_results.json")
    # print(f"image_paths = {image_paths}")