import psycopg2

DATABASE_URL = "postgresql://postgres:1234@ec2-52-194-194-37.ap-northeast-1.compute.amazonaws.com:5434/image"

def check_database_stats():
    """
    檢查資料庫中的統計資訊
    """
    print("Database Statistics")
    print("=" * 60)
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # 總記錄數
        cursor.execute("SELECT COUNT(*) FROM image_library;")
        total_count = cursor.fetchone()[0]
        print(f"Total records: {total_count}")
        
        # 按 sub_theme 統計
        cursor.execute("""
            SELECT sub_theme, COUNT(*) 
            FROM image_library 
            GROUP BY sub_theme 
            ORDER BY COUNT(*) DESC;
        """)
        
        theme_stats = cursor.fetchall()
        
        print(f"\nRecords by sub_theme:")
        print("-" * 40)
        
        for theme, count in theme_stats:
            print(f"  {theme:<20} | {count:>3} records")
        
        # 按 theme 統計（應該全部都是 lens_quant）
        cursor.execute("""
            SELECT theme, COUNT(*) 
            FROM image_library 
            GROUP BY theme;
        """)
        
        main_theme_stats = cursor.fetchall()
        
        print(f"\nRecords by main theme:")
        print("-" * 40)
        
        for theme, count in main_theme_stats:
            print(f"  {theme:<20} | {count:>3} records")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_database_stats()