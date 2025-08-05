import psycopg2

DATABASE_URL = "postgresql://postgres:1234@ec2-52-194-194-37.ap-northeast-1.compute.amazonaws.com:5434/image"

def check_image_library_table():
    print("Checking image_library table structure...")
    print("-" * 60)
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # 檢查 image_library 表結構
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'image_library'
            ORDER BY ordinal_position;
        """)
        columns = cursor.fetchall()
        
        if columns:
            print("✅ image_library table structure:")
            print("Column Name          | Data Type    | Nullable | Default")
            print("-" * 60)
            for col_name, data_type, is_nullable, col_default in columns:
                default_str = str(col_default)[:20] if col_default else "None"
                print(f"{col_name:<20} | {data_type:<12} | {is_nullable:<8} | {default_str}")
        else:
            print("❌ image_library table not found")
            return False
        
        # 檢查表中是否有資料
        print(f"\nChecking existing data...")
        cursor.execute("SELECT COUNT(*) FROM image_library;")
        count = cursor.fetchone()[0]
        print(f"Current record count: {count}")
        
        # 如果有資料，顯示幾筆範例
        if count > 0:
            cursor.execute("SELECT * FROM image_library LIMIT 3;")
            rows = cursor.fetchall()
            print(f"\nSample records:")
            for i, row in enumerate(rows, 1):
                print(f"Record {i}: {row}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    check_image_library_table()