#!/usr/bin/env python3
"""
Test script to add file_path for a few items
"""

import json
import sys
import os

# Add the current directory to path to import the function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from add_file_paths import get_file_info_from_comfy_history

def test_single_task():
    # Test with one task_id
    task_id = "3031619e-dec3-4487-ae3a-16d36d308083"
    print(f"Testing task_id: {task_id}")
    
    file_info = get_file_info_from_comfy_history(task_id)
    if file_info:
        print(f"✅ Success: {file_info}")
    else:
        print("❌ Failed to retrieve file info")

if __name__ == "__main__":
    test_single_task()