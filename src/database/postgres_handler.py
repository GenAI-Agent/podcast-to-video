import json
import os
import glob
import re
from typing import List, Dict, Optional
import psycopg2
from psycopg2.extras import execute_values
import uuid

# Database configuration
DATABASE_URL = "postgresql://postgres:1234@ec2-52-194-194-37.ap-northeast-1.compute.amazonaws.com:5434/image"

def extract_image_info_from_task_id(task_id: str, file_name: str) -> Dict[str, str]:
    """
    根據 task_id 和 file_name 生成預期的圖片檔案名稱
    
    實際圖片路徑格式: C:\\Users\\x7048\\Documents\\ComfyUI\\output\\batch_retail_savings\\batch_retail_savings_2025-07-31_17-30-53-0001.png
    file_name 格式: batch_risk_vs_reward
    
    返回:
    {
        'expected_filename': 'batch_retail_savings_2025-07-31_17-30-53-0001.png',
        'sub_theme': 'retail_savings',
        'relative_path': 'batch_retail_savings/batch_retail_savings_2025-07-31_17-30-53-0001.png'
    }
    """
    # 從 file_name 提取 sub_theme
    sub_theme = file_name.replace('batch_', '')
    
    # 根據 task_id 的順序生成編號
    # 注意：這裡假設 task_id 是按順序的，實際可能需要根據 JSON 中的順序來確定
    # 暫時返回預期的格式，實際檔案需要根據實際情況匹配
    
    return {
        'sub_theme': sub_theme,
        'expected_pattern': f"batch_{sub_theme}_*-*.png",
        'relative_path': f"batch_{sub_theme}/",  # 子目錄路徑
    }

def prepare_batch_data(json_file_path: str) -> List[Dict]:
    """
    準備要插入到資料庫的批次資料
    
    JSON 欄位對應到資料庫欄位：
    - task_id → id
    - file_name → name  
    - file_path → file_path
    - prompt → prompt
    - tags → description
    
    Args:
        json_file_path: JSON 檔案路徑
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
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
            
            # JSON 中的 key 作為 sub_theme，所有 theme 都是 lens_quant
            sub_theme = theme  # JSON 的 key (如 REAL_ESTATE, DAY_TRADING) 作為 sub_theme
            
            # 準備資料記錄（直接使用 JSON 中的資料）
            record = {
                'id': task_id,  # 直接使用 task_id 作為 id
                'name': file_name or f"{theme}_{task_id[:8]}",  # 使用 file_name 作為 name
                'file_path': file_path or '',  # 使用 JSON 中的 file_path
                'prompt': prompt,
                'description': tags,  # 使用 tags 作為 description
                'theme': 'lens_quant',  # 所有記錄的 theme 都是 lens_quant
                'sub_theme': sub_theme,  # JSON 的 key 作為 sub_theme
                'status': 'active'
            }
            
            batch_data.append(record)
    
    return batch_data

def batch_insert_to_postgres(batch_data: List[Dict]):
    """
    批量插入資料到 PostgreSQL
    """
    if not batch_data:
        print("No data to insert")
        return
    
    conn = None
    cursor = None
    
    try:
        # 建立資料庫連接
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # 準備插入語句
        insert_query = """
            INSERT INTO image_library (id, name, file_path, prompt, description, theme, sub_theme, status)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                file_path = EXCLUDED.file_path,
                prompt = EXCLUDED.prompt,
                description = EXCLUDED.description,
                theme = EXCLUDED.theme,
                sub_theme = EXCLUDED.sub_theme,
                status = EXCLUDED.status
        """
        
        # 準備資料 tuple
        values = [
            (
                record['id'],
                record['name'],
                record['file_path'],
                record['prompt'],
                record['description'],
                record['theme'],
                record['sub_theme'],
                record['status']
            )
            for record in batch_data
        ]
        
        # 執行批量插入
        execute_values(cursor, insert_query, values)
        
        # 提交變更
        conn.commit()
        
        print(f"Successfully inserted/updated {len(batch_data)} records")
        
    except Exception as e:
        print(f"Error during database operation: {e}")
        if conn:
            conn.rollback()
        raise
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def create_task_id_mapping(json_file_path: str) -> Dict[str, str]:
    """
    創建 task_id 到實際檔案名稱的對應關係
    需要您提供實際的檔案列表來完成對應
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    mapping = {}
    
    print("Task ID to File Mapping Helper")
    print("=" * 80)
    
    for theme, items in data.items():
        print(f"\nTheme: {theme}")
        for idx, item in enumerate(items, 1):
            task_id = item.get('task_id')
            file_name = item.get('file_name')
            
            if task_id and file_name:
                sub_theme = file_name.replace('batch_', '')
                expected_pattern = f"batch_{sub_theme}_*-{idx:04d}.png"
                
                print(f"  [{idx}] Task ID: {task_id}")
                print(f"      Expected pattern: {expected_pattern}")
                
                mapping[task_id] = {
                    'index': idx,
                    'sub_theme': sub_theme,
                    'pattern': expected_pattern
                }
    
    return mapping

def process_all_updated_json_files():
    """
    處理所有 _updated.json 檔案並匯入到 PostgreSQL
    """
    # 尋找所有 _updated.json 檔案
    json_files = glob.glob("data/json/*_updated.json")
    
    if not json_files:
        print("No _updated.json files found to process")
        return
    
    print("Found JSON files to process:")
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
            batch_data = prepare_batch_data(json_file)
            all_batch_data.extend(batch_data)
            print(f"  ✅ Prepared {len(batch_data)} records")
        except Exception as e:
            print(f"  ❌ Error processing {json_file}: {e}")
    
    if all_batch_data:
        print(f"\nTotal records to insert: {len(all_batch_data)}")
        
        # 自動進行 PostgreSQL 匯入
        print("\nStarting PostgreSQL import...")
        try:
            batch_insert_to_postgres(all_batch_data)
            print("\n✅ Import completed successfully!")
        except Exception as e:
            print(f"\n❌ Import failed: {e}")
    else:
        print("No data to import")

if __name__ == "__main__":
    process_all_updated_json_files()