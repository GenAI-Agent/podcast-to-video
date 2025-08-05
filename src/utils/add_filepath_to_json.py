import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

def extract_order_from_json(json_file_path: str) -> Dict[str, int]:
    """
    從 JSON 檔案中提取每個 theme 中 task_id 的順序
    返回 task_id 到順序編號的映射
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    task_order_mapping = {}
    
    for theme, items in data.items():
        # 為每個主題的項目分配順序編號
        for idx, item in enumerate(items, 1):
            task_id = item.get('task_id')
            if task_id:
                task_order_mapping[task_id] = {
                    'order': idx,
                    'theme': theme,
                    'number': f"{idx:04d}"  # 格式化為 0001, 0002 等
                }
    
    return task_order_mapping

def generate_file_path(task_id: str, file_name: str, task_order_mapping: Dict, 
                      base_timestamp: str = "2025-07-31_17-30-53") -> str:
    """
    根據 task_id 和 file_name 生成檔案路徑
    
    Args:
        task_id: 任務ID
        file_name: 檔案名稱 (如 batch_retail_savings)
        task_order_mapping: task_id 到順序的映射
        base_timestamp: 基礎時間戳
    
    Returns:
        生成的檔案路徑
    """
    if task_id not in task_order_mapping:
        return ""
    
    order_info = task_order_mapping[task_id]
    number = order_info['number']
    
    # 從 file_name 提取 sub_theme
    sub_theme = file_name.replace('batch_', '')
    
    # 構建檔案名稱
    filename = f"{file_name}_{base_timestamp}-{number}.png"
    
    # 構建完整路徑 (Windows 格式)
    # 注意：這裡使用你提供的路徑格式
    file_path = f"C:\\Users\\x7048\\Documents\\ComfyUI\\output\\{file_name}\\{filename}"
    
    return file_path

def add_filepath_to_json(json_file_path: str, output_file: str = None, 
                        base_timestamp: str = "2025-07-31_17-30-53"):
    """
    為 JSON 檔案中的每個項目添加 file_path
    
    Args:
        json_file_path: 輸入的 JSON 檔案路徑
        output_file: 輸出檔案路徑（如果為 None，則生成預設名稱）
        base_timestamp: 基礎時間戳（根據實際圖片生成時間調整）
    """
    print(f"Processing JSON file: {json_file_path}")
    
    # 讀取 JSON 檔案
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取 task_id 的順序
    task_order_mapping = extract_order_from_json(json_file_path)
    print(f"Found {len(task_order_mapping)} task IDs")
    
    # 更新每個項目
    updated_count = 0
    for theme, items in data.items():
        print(f"\nProcessing theme: {theme}")
        for item in items:
            task_id = item.get('task_id')
            file_name = item.get('file_name')
            
            if task_id and file_name:
                file_path = generate_file_path(task_id, file_name, task_order_mapping, base_timestamp)
                if file_path:
                    item['file_path'] = file_path
                    updated_count += 1
                    
                    # 顯示前幾個作為範例
                    if updated_count <= 5:
                        print(f"  Task ID: {task_id}")
                        print(f"  File path: {file_path}")
    
    # 生成輸出檔案名稱
    if output_file is None:
        base_name = os.path.splitext(json_file_path)[0]
        output_file = f"{base_name}_with_filepath.json"
    
    # 寫入更新後的 JSON
    print(f"\nWriting updated JSON to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSummary: Updated {updated_count} items with file paths")
    return output_file

def create_filepath_mapping_csv(json_file_path: str, output_csv: str = "filepath_mapping.csv"):
    """
    創建一個 CSV 檔案，顯示 task_id 到檔案路徑的對應關係
    """
    import csv
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    task_order_mapping = extract_order_from_json(json_file_path)
    
    rows = []
    for theme, items in data.items():
        for item in items:
            task_id = item.get('task_id')
            file_name = item.get('file_name')
            
            if task_id and file_name:
                order_info = task_order_mapping.get(task_id, {})
                file_path = generate_file_path(task_id, file_name, task_order_mapping)
                
                rows.append({
                    'theme': theme,
                    'order': order_info.get('order', ''),
                    'task_id': task_id,
                    'file_name': file_name,
                    'file_path': file_path
                })
    
    # 寫入 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['theme', 'order', 'task_id', 'file_name', 'file_path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"CSV mapping saved to: {output_csv}")
    return output_csv

if __name__ == "__main__":
    # 處理所有 JSON 檔案
    json_files = [
        "/home/fluxmind/batch_image/batch_image_data_20250801_100459_with_tags.json",
        "/home/fluxmind/batch_image/batch_image_data_20250801_105709_with_tags.json",
        "/home/fluxmind/batch_image/batch_image_data_20250801_113950_with_tags.json",
        "/home/fluxmind/batch_image/batch_image_data_20250801_123913_with_tags.json",
        "/home/fluxmind/batch_image/batch_image_data_20250801_133312_with_tags.json",
        "/home/fluxmind/batch_image/batch_image_data_20250801_155450_with_tags.json",
        "/home/fluxmind/batch_image/batch_image_data_20250801_170613_with_tags.json"
    ]
    
    # 選擇要處理的檔案
    # 這裡我們先處理最新的檔案作為範例
    json_file = json_files[-1]  # 使用最新的檔案
    
    print("=" * 80)
    print("Batch Image File Path Generator")
    print("=" * 80)
    
    # 生成 CSV 映射檔案
    print("\n1. Creating CSV mapping file...")
    create_filepath_mapping_csv(json_file)
    
    # 更新 JSON 檔案
    print("\n2. Updating JSON file with file paths...")
    output_file = add_filepath_to_json(json_file)
    
    print("\nDone!")