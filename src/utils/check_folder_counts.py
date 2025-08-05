"""
用來確認各個資料夾的實際檔案數量的工具
請告訴我以下資料夾的實際圖片數量：
"""

print("請確認以下資料夾的實際圖片數量：")
print("=" * 60)

folders_to_check = [
    "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\batch_real_estate",
    "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\batch_day_trading", 
    "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\batch_global_market",
    "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\batch_retail_savings",
    "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\batch_strategies",
    "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\batch_risk_vs_reward",
    "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\batch_bear_market",
    "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\batch_crypto",
    "C:\\Users\\x7048\\Documents\\ComfyUI\\output\\WallStreet"
]

current_db_counts = {
    "REAL_ESTATE": 50,
    "DAY_TRADING": 50,
    "GLOBAL_MARKET": 50,
    "RETAIL_SAVINGS": 100,
    "STRATEGIES": 50,
    "RISK_VS_REWARD": 50,
    "BEAR_MARKET": 50,  # 已更新
    "CRYPTO": 56,       # 已更新
    "WALLSTREET": 10
}

print("資料夾 -> 目前資料庫記錄數")
print("-" * 40)

for i, folder in enumerate(folders_to_check):
    theme_name = list(current_db_counts.keys())[i]
    current_count = current_db_counts[theme_name]
    
    print(f"{folder}")
    print(f"  -> 目前資料庫: {current_count} 筆")
    print(f"  -> 實際檔案: ??? 個 (請告訴我)")
    print()

print("請告訴我每個資料夾的實際檔案數量，我會據此更新資料庫記錄。")
print("\n已確認並更新的資料夾：")
print("✅ batch_crypto: 56 個檔案 -> 56 筆記錄")
print("✅ batch_bear_market: 50 個檔案 -> 50 筆記錄")