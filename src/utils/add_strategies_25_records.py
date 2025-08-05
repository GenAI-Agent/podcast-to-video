import json
import uuid
from datetime import datetime
import psycopg2
from import_to_postgres_v2 import batch_insert_to_postgres

DATABASE_URL = "postgresql://postgres:1234@ec2-52-194-194-37.ap-northeast-1.compute.amazonaws.com:5434/image"

def generate_additional_strategies_records():
    """
    為 STRATEGIES 生成額外的 25 筆記錄（編號 51-75）
    """
    
    # Strategies 相關的 prompt 主題
    strategies_prompts = [
        "Advanced algorithmic trading strategy visualization with complex mathematical formulas",
        "Portfolio diversification strategy displayed as interconnected investment nodes",
        "Risk management framework diagram with multiple safety checkpoints",
        "Value investing strategy blueprint with fundamental analysis charts",
        "Technical analysis strategy combining multiple indicator overlays",
        "Options trading strategy wheel with profit and loss scenarios",
        "Day trading strategy setup with multiple timeframe analysis",
        "Long-term investment strategy timeline spanning decades",
        "Hedge fund strategy visualization with market neutral positions",
        "Cryptocurrency trading strategy with volatility indicators",
        "Asset allocation strategy pie chart with rebalancing triggers",
        "Momentum trading strategy with trend following signals",
        "Mean reversion strategy with statistical arbitrage opportunities",
        "Dollar-cost averaging strategy over market cycles",
        "Swing trading strategy with support and resistance levels",
        "Growth investing strategy focusing on emerging sectors",
        "Dividend growth strategy with compound interest visualization",
        "Market timing strategy with economic cycle indicators",
        "Pairs trading strategy with correlation analysis",
        "Sector rotation strategy based on economic phases",
        "Quantitative trading strategy with backtesting results",
        "ESG investing strategy with sustainability metrics",
        "International diversification strategy across global markets",
        "Real estate investment strategy with market analysis",
        "Retirement planning strategy with lifecycle funds"
    ]
    
    additional_records = []
    timestamp = "2025-07-31_17-30-53"  # 使用標準時間戳
    
    # 生成編號 51-75 的記錄
    for i in range(51, 76):  # 51 到 75，共 25 筆
        # 生成唯一的 task_id
        task_id = str(uuid.uuid4())
        
        # 選擇 prompt（循環使用）
        prompt_index = (i - 51) % len(strategies_prompts)
        base_prompt = strategies_prompts[prompt_index]
        
        # 構建完整的 prompt
        full_prompt = f'''
{{
  "prompt": {{
    "description": "{base_prompt}. Professional financial strategy visualization with analytical depth.",
    "camera": {{
      "type": "Full-frame DSLR",
      "lens": "50mm f/1.8 prime lens",
      "settings": {{
        "aperture": "f/2.8",
        "shutter_speed": "1/160",
        "ISO": "200"
      }}
    }},
    "lighting": {{
      "type": "Professional studio lighting",
      "mood": "Clean and analytical with strategic emphasis"
    }},
    "style": {{
      "quality": "High resolution, professional",
      "mood": "Strategic, analytical, forward-thinking"
    }}
  }}
}}
'''
        
        # 生成檔案資訊
        number = f"{i:04d}"
        file_name = f"batch_strategies_{timestamp}-{number}"
        file_path = f"C:\\Users\\x7048\\Documents\\ComfyUI\\output\\batch_strategies\\{file_name}.png"
        
        # 模擬 tags
        tags = "no humans, strategy, analysis, chart, graph, financial planning, "
        
        # 構建項目（準備插入資料庫的格式）
        record = {
            'id': task_id,
            'name': file_name,
            'file_path': file_path,
            'prompt': full_prompt,
            'description': tags,
            'theme': 'lens_quant',
            'sub_theme': 'STRATEGIES',
            'status': 'active'
        }
        
        additional_records.append(record)
    
    return additional_records

def add_strategies_records_to_database():
    """
    將額外的 25 筆 STRATEGIES 記錄添加到資料庫
    """
    print("Generating additional 25 STRATEGIES records (51-75)...")
    
    # 生成額外的記錄
    additional_records = generate_additional_strategies_records()
    
    print(f"Generated {len(additional_records)} additional records")
    print("Range: 0051 to 0075")
    
    # 顯示範例
    print(f"\nFirst 3 additional records:")
    for i, record in enumerate(additional_records[:3]):
        print(f"\n[{i+51}] Task ID: {record['id']}")
        print(f"    File: {record['name']}")
        print(f"    Path: {record['file_path']}")
    
    # 插入到資料庫
    print(f"\nInserting {len(additional_records)} additional records into database...")
    
    try:
        # 直接插入新記錄（不刪除現有的）
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # 準備插入語句
        insert_query = """
            INSERT INTO image_library (id, name, file_path, prompt, description, theme, sub_theme, status)
            VALUES %s
        """
        
        # 準備資料 tuple
        from psycopg2.extras import execute_values
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
            for record in additional_records
        ]
        
        # 執行批量插入
        execute_values(cursor, insert_query, values)
        
        # 提交變更
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Successfully added {len(additional_records)} additional STRATEGIES records")
        
        # 檢查最終數量
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM image_library WHERE sub_theme = 'STRATEGIES';")
        final_count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        print(f"✅ Total STRATEGIES records now: {final_count}")
        
    except Exception as e:
        print(f"❌ Error adding records: {e}")

if __name__ == "__main__":
    add_strategies_records_to_database()