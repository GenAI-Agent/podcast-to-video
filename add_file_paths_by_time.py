#!/usr/bin/env python3
"""
Script to add file_path field to JSON files by matching with files in ComfyUI output folders
based on file creation time order
"""

import json
import os
import time
from typing import Dict, Any, List, Tuple
from pathlib import Path

def wsl_to_windows_path(wsl_path: str) -> str:
    """
    Convert WSL path to Windows path format
    
    Args:
        wsl_path (str): Path in WSL format (e.g., /mnt/c/Users/...)
        
    Returns:
        str: Path in Windows format (e.g., C:\\Users\\...)
    """
    if wsl_path.startswith("/mnt/c/"):
        # Convert /mnt/c/ to C:\
        windows_path = wsl_path.replace("/mnt/c/", "C:\\")
        # Replace forward slashes with backslashes
        windows_path = windows_path.replace("/", "\\")
        return windows_path
    return wsl_path

def get_files_by_creation_time(folder_path: str) -> List[Tuple[str, str, float]]:
    """
    Get all image files in a folder sorted by creation time
    
    Args:
        folder_path (str): Path to the folder containing images
        
    Returns:
        List[Tuple[str, str, float]]: List of tuples (file_name_without_ext, full_path, creation_time)
    """
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist")
        return []
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
    files_with_time = []
    
    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Check if it's a file and has image extension
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(file_name)[1].lower()
                if file_ext in image_extensions:
                    # Get creation time (or modification time if creation time is not available)
                    try:
                        creation_time = os.path.getctime(file_path)
                    except:
                        creation_time = os.path.getmtime(file_path)
                    
                    file_name_without_ext = os.path.splitext(file_name)[0]
                    files_with_time.append((file_name_without_ext, file_path, creation_time))
        
        # Sort by creation time (oldest first)
        files_with_time.sort(key=lambda x: x[2])
        
        return files_with_time
        
    except Exception as e:
        print(f"Error reading folder {folder_path}: {str(e)}")
        return []

def get_category_folder_mapping() -> Dict[str, str]:
    """
    Define mapping between JSON categories and their corresponding output folders
    
    Returns:
        Dict[str, str]: Mapping of category names to folder paths
    """
    # Check if we're running in WSL
    import platform
    system = platform.system().lower()
    
    if system == "linux" and "microsoft" in platform.uname().release.lower():
        # We're in WSL, use the /mnt/c/ path
        base_path = "/mnt/c/Users/x7048/Documents/ComfyUI/output"
    else:
        # We're on Windows, use the Windows path
        base_path = r"C:\Users\x7048\Documents\ComfyUI\output"
    
    # First, let's check what folders actually exist
    available_folders = []
    if os.path.exists(base_path):
        try:
            available_folders = [f for f in os.listdir(base_path) 
                               if os.path.isdir(os.path.join(base_path, f))]
        except:
            pass
    
    # Create mapping based on available folders
    mapping = {}
    
    # Try to match categories with existing folders
    category_patterns = {
        "CHINESE_KINGDOMS": ["chinese_kingdoms", "chinese", "kingdoms", "china"],
        "DRAMA": ["drama", "dramatic", "theater"],
        "FANTASY_ADVENTURE": ["fantasy", "fantasy_adventure", "adventure", "fantasy_image"],
        "ROMANCE": ["romance", "romantic", "love"],
        "SPOOKY_STORY": ["spooky", "spooky_story", "horror", "scary"]
    }
    
    for category, patterns in category_patterns.items():
        matched = False
        for folder in available_folders:
            folder_lower = folder.lower()
            for pattern in patterns:
                if pattern in folder_lower:
                    mapping[category] = os.path.join(base_path, folder)
                    matched = True
                    break
            if matched:
                break
        
        # If no match found, use the first pattern as default
        if not matched:
            mapping[category] = os.path.join(base_path, patterns[0])
    
    return mapping

def add_file_paths_by_time(json_file_path: str, output_file_path: str = None) -> None:
    """
    Add file_path field to all items in a JSON file by matching with files based on creation time
    
    Args:
        json_file_path (str): Path to the input JSON file
        output_file_path (str): Path to save the updated JSON file (optional)
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If no output file specified, create one with "_with_paths" suffix
        if output_file_path is None:
            base_name = os.path.splitext(json_file_path)[0]
            output_file_path = f"{base_name}_with_paths.json"
        
        # Get folder mapping
        folder_mapping = get_category_folder_mapping()
        
        # Debug: Show folder mapping
        print(f"\n📁 Folder mapping:")
        
        # Get the base path used in folder_mapping function
        import platform
        system = platform.system().lower()
        if system == "linux" and "microsoft" in platform.uname().release.lower():
            base_path = "/mnt/c/Users/x7048/Documents/ComfyUI/output"
            print(f"🐧 Running in WSL, using path: {base_path}")
        else:
            base_path = r"C:\Users\x7048\Documents\ComfyUI\output"
            print(f"🪟 Running on Windows, using path: {base_path}")
        
        if os.path.exists(base_path):
            available_folders = [f for f in os.listdir(base_path) 
                               if os.path.isdir(os.path.join(base_path, f))]
            print(f"Available folders in {base_path}:")
            for folder in available_folders:
                print(f"  - {folder}")
        else:
            print(f"❌ Base path does not exist: {base_path}")
        
        print(f"\nMapping used:")
        for category, path in folder_mapping.items():
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"  {category} → {path} {exists}")
        
        updated_items = 0
        total_items = 0
        
        # Process each category in the JSON
        for category, items in data.items():
            print(f"\n=== Processing category: {category} ===")
            
            if category not in folder_mapping:
                print(f"Warning: No folder mapping found for category '{category}', skipping...")
                continue
            
            folder_path = folder_mapping[category]
            print(f"Looking for files in: {folder_path}")
            
            # Get files sorted by creation time
            files_with_time = get_files_by_creation_time(folder_path)
            
            if not files_with_time:
                print(f"No image files found in {folder_path}")
                continue
            
            print(f"Found {len(files_with_time)} files in folder")
            print(f"Found {len(items)} items in JSON category")
            
            # Match files to JSON items based on order
            for i, item in enumerate(items):
                total_items += 1
                
                if i < len(files_with_time):
                    file_name_without_ext, full_path, creation_time = files_with_time[i]
                    
                    # Convert path to Windows format for storage (more universal)
                    windows_path = wsl_to_windows_path(full_path)
                    
                    # Update the JSON item
                    item['file_name'] = file_name_without_ext
                    item['file_path'] = windows_path
                    
                    # Convert creation time to readable format for logging
                    readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time))
                    
                    print(f"  [{i+1:2d}] Matched: {file_name_without_ext} (created: {readable_time})")
                    print(f"       Path: {windows_path}")
                    updated_items += 1
                    
                else:
                    print(f"  [{i+1:2d}] No corresponding file found for item {i+1}")
        
        # Save the updated data
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== Summary ===")
        print(f"Total items processed: {total_items}")
        print(f"Items updated with file_path: {updated_items}")
        print(f"Updated data saved to: {output_file_path}")
        
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def main():
    """Main function to process all JSON files with GPT tags"""
    
    # List of JSON files to process
    json_files = [
        "chinese_kingdoms_data_20250805_103731_with_gpt_tags.json",
        "drama_data_20250805_144158_with_gpt_tags.json", 
        "fantasy_image_data_20250805_091354_with_gpt_tags.json",
        "romance_data_20250805_133414_with_gpt_tags.json",
        "spooky_story_data_20250805_113814_with_gpt_tags.json"
    ]
    
    print("Starting to add file_path fields to JSON files based on file creation time...")
    print("This script will match JSON items with files in their corresponding folders")
    print("based on the chronological order of file creation.\n")
    
    # Ask for confirmation
    response = input("Do you want to continue? (y/n): ").strip().lower()
    if response != 'y' and response != 'yes':
        print("Operation cancelled.")
        return
    
    processed_files = 0
    
    for json_file in json_files:
        if os.path.exists(json_file):
            print(f"\n{'='*80}")
            print(f"Processing: {json_file}")
            print(f"{'='*80}")
            
            # Create output filename
            base_name = os.path.splitext(json_file)[0]
            # Remove '_with_gpt_tags' suffix and add '_final'
            if base_name.endswith('_with_gpt_tags'):
                base_name = base_name[:-14]  # Remove '_with_gpt_tags'
            output_file = f"{base_name}_final.json"
            
            add_file_paths_by_time(json_file, output_file)
            processed_files += 1
        else:
            print(f"Warning: File {json_file} not found, skipping...")
    
    print(f"\n{'='*80}")
    print(f"Processing complete! Successfully processed {processed_files} files.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()