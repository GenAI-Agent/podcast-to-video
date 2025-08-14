# langchain、model
from langchain_openai import AzureChatOpenAI    
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# system
import sys
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
# 禁用SSL警告（可選）
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import base64
from typing import Dict, Any, Optional, List
import random
import asyncio
from dotenv import load_dotenv
from datetime import datetime
import threading
# 專案項目
# 獲取專案根目錄的路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from generators.prompts.image_generator import GENERIC_IMAGE_PROMPT, fantasy_adventure_scenarios, magical_quest_scenarios, epic_battle_scenarios, astrology_scenarios, GENERIC_IMAGE_PROMPT

load_dotenv()
api_base = os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

def safe_request(method: str, url: str, **kwargs) -> requests.Response:
    """
    安全的HTTP請求，自動處理SSL證書問題
    
    Args:
        method (str): HTTP方法 ('GET', 'POST', etc.)
        url (str): 請求URL
        **kwargs: 其他requests參數
    
    Returns:
        requests.Response: HTTP響應
    """
    # 設置默認超時和SSL配置
    kwargs.setdefault('timeout', 30)
    kwargs.setdefault('verify', False)  # 禁用SSL驗證
    
    try:
        if method.upper() == 'GET':
            return requests.get(url, **kwargs)
        elif method.upper() == 'POST':
            return requests.post(url, **kwargs)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
    except requests.exceptions.SSLError as e:
        print(f"SSL錯誤，使用session重試: {e}")
        # 創建忽略SSL的session
        session = requests.Session()
        session.verify = False
        
        if method.upper() == 'GET':
            return session.get(url, **kwargs)
        elif method.upper() == 'POST':
            return session.post(url, **kwargs)
    except Exception as e:
        print(f"請求失敗: {e}")
        raise

def generate_image_prompt_fun(description:str) -> str:
    """
    生成圖片prompt
    Args:
        description (str): 描述
    Returns:
        str: image_prompt
    """
    # system_prompt1 = image_first_prompt
    # system_prompt2 = image_second_prompt
    llm = AzureChatOpenAI(
        azure_endpoint=api_base,
        api_key=api_key,
        azure_deployment="gpt-4o-testing",
        api_version="2025-01-01-preview",
        temperature=0.6,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    generate_first_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", GENERIC_IMAGE_PROMPT),
            ("user", "Specific description: {description}")
        ]
    )
    generate_first_prompt_chain = generate_first_prompt_template | llm | StrOutputParser()
    prompt = generate_first_prompt_chain.invoke({"description": description })
    if prompt:
        prompt = prompt.replace("```json", "").replace("```", "")
    return prompt

def call_image_request_function(prompt: str, path_name:str) -> Optional[str]:
    """
    使用 ComfyUI 生成圖片
    
    參數:
        prompt (str): prompt_input
        path_name (str): 路徑名稱
    返回:
        Optional[str]: 生成的圖片的 base64 字符串，失敗則返回 None
    """
    try:
        # ComfyUI 的 API 端點

        # COMFY_URL = "http://localhost:8000/" # TODO: 改成要用的url，格式：https://da07-185-219-141-17.ngrok-free.app/api/prompt
        COMFY_URL = "https://image-server.ask-lens.ai/api/prompt"
        if COMFY_URL:
            workflow_path = [
                "workflows/KreaGen.json",
            ]
            # 隨機選擇一個 workflow_path
            selected_path = random.choice(workflow_path)
            with open(f"{selected_path}", "r", encoding='utf-8') as file:
                workflow = json.load(file)

            # Process workflow nodes to replace placeholders
            for node in workflow.values():
                # Replace prompt text in CLIPTextEncode nodes
                if node.get("class_type") == "CLIPTextEncode":
                    if "inputs" in node and "text" in node["inputs"]:
                        # Replace the entire text with the generated prompt
                        node["inputs"]["text"] = prompt
                
                # Update SaveImageExtended to use custom folder name and simpler filename
                if node.get("class_type") == "SaveImageExtended":
                    if "inputs" in node:
                        # Set custom folder name
                        if "foldername_keys" in node["inputs"]:
                            node["inputs"]["foldername_keys"] = path_name
                        # Simplify filename to avoid length issues
                        if "filename_prefix" in node["inputs"]:
                            node["inputs"]["filename_prefix"] = f"{path_name}_%F_%H-%M-%S"
                        # Disable tagger-based filename keys to prevent tags in filename
                        if "filename_keys" in node["inputs"]:
                            node["inputs"]["filename_keys"] = ""  # Remove tagger reference from filename

                # Support core SaveImage node by setting subfolder/filename_prefix
                if node.get("class_type") == "SaveImage":
                    if "inputs" in node:
                        if "subfolder" in node["inputs"]:
                            node["inputs"]["subfolder"] = path_name
                        if "filename_prefix" in node["inputs"]:
                            node["inputs"]["filename_prefix"] = f"{path_name}/{path_name}_%F_%H-%M-%S"
            
            print(f"Processing workflow with prompt length: {len(prompt)} characters")
            print(f"Output folder: {path_name}")
            # 發送請求執行 workflow
            response = safe_request('POST', COMFY_URL, json={"prompt": workflow})
            
            if response.status_code != 200:
                print(f"錯誤: ComfyUI 請求失敗，狀態碼: {response.status_code}, {response.text}")
                return None
            
            # 獲取排隊ID
            prompt_id = response.json().get('prompt_id')
            if not prompt_id:
                print("錯誤: 未能獲取prompt_id")
                return None
            return prompt_id
        
        else:
            return "test_prompt_id"
            
        # # 等待圖片生成完成
        # history_url = f"https://0691-94-140-8-49.ngrok-free.app/history/{prompt_id}"
        # max_attempts = 30  # 最大等待次數
        # attempt = 0
        
        # while attempt < max_attempts:
        #     history_response = requests.get(history_url)
        #     if history_response.status_code == 200:
        #         history_data = history_response.json()
        #         if history_data.get('status', {}).get('completed', False):
        #             print("圖片生成完成")
        #             return "success"  # 或返回其他所需信息
            
        #     await asyncio.sleep(2)  # 等待2秒後再次檢查
        #     attempt += 1
        
        # print("圖片生成超時")
        # return None
    
    except Exception as e:
        print(f"發送生成圖片請求時發生錯誤: {str(e)}")
        return None

def check_image_generation_status(prompt_id: str) -> Dict[str, Any]:
    """
    檢查ComfyUI圖片生成狀態
    
    Args:
        prompt_id (str): ComfyUI返回的prompt ID
        
    Returns:
        Dict[str, Any]: 狀態信息，包含 'status', 'progress', 'data'
    """
    try:
        # 檢查任務狀態的URL
        queue_url = f"https://image-server.ask-lens.ai/queue"
        history_url = f"https://image-server.ask-lens.ai/history/{prompt_id}"
        
        # 首先檢查是否在隊列中
        queue_response = safe_request('GET', queue_url)
        
        if queue_response.status_code == 200:
            queue_data = queue_response.json()
            
            # 檢查是否在執行隊列中
            queue_running = queue_data.get('queue_running', [])
            queue_pending = queue_data.get('queue_pending', [])
            
            # 檢查執行隊列，處理不同的數據格式
            for item in queue_running:
                if isinstance(item, dict):
                    if item.get('prompt_id') == prompt_id:
                        return {
                            'status': 'processing',
                            'progress': 'running',
                            'data': None,
                            'message': 'Image is currently being generated'
                        }
                elif isinstance(item, list) and len(item) >= 2:
                    # 某些情況下格式可能是 [number, {'prompt_id': ...}]
                    if isinstance(item[1], dict) and item[1].get('prompt_id') == prompt_id:
                        return {
                            'status': 'processing',
                            'progress': 'running',
                            'data': None,
                            'message': 'Image is currently being generated'
                        }
            
            # 檢查待處理隊列
            for item in queue_pending:
                if isinstance(item, dict):
                    if item.get('prompt_id') == prompt_id:
                        return {
                            'status': 'queued',
                            'progress': 'pending',
                            'data': None,
                            'message': 'Image is queued for generation'
                        }
                elif isinstance(item, list) and len(item) >= 2:
                    if isinstance(item[1], dict) and item[1].get('prompt_id') == prompt_id:
                        return {
                            'status': 'queued',
                            'progress': 'pending',
                            'data': None,
                            'message': 'Image is queued for generation'
                        }
        
        # 檢查歷史記錄中是否已完成
        history_response = safe_request('GET', history_url)
            
        if history_response.status_code == 200:
            history_data = history_response.json()
            
            if prompt_id in history_data:
                prompt_data = history_data[prompt_id]
                outputs = prompt_data.get('outputs', {})
                
                if outputs:
                    # 任務已完成，提取資訊
                    result = {
                        'tags': None,
                        'file_name': None,
                        'file_path': None
                    }
                    
                    # 尋找WD14 Tagger輸出
                    if '39' in outputs:
                        tagger_output = outputs['39']
                        if 'tags' in tagger_output:
                            tags = tagger_output['tags'][0] if isinstance(tagger_output['tags'], list) else tagger_output['tags']
                            result['tags'] = tags
                    
                    # 尋找SaveImage節點輸出
                    for node_id, output in outputs.items():
                        if 'images' in output:
                            images = output['images']
                            if images and len(images) > 0:
                                image_info = images[0]
                                file_name = image_info.get('filename', '')
                                if file_name:
                                    result['file_name'] = os.path.splitext(file_name)[0]
                                    subfolder = image_info.get('subfolder', '')
                                    if subfolder:
                                        result['file_path'] = f"C:\\Users\\x7048\\Documents\\ComfyUI\\output\\{subfolder}\\{file_name}"
                                    else:
                                        result['file_path'] = f"C:\\Users\\x7048\\Documents\\ComfyUI\\output\\{file_name}"
                                    break
                    
                    return {
                        'status': 'completed',
                        'progress': 'finished',
                        'data': result,
                        'message': 'Image generation completed successfully'
                    }
                else:
                    # 在歷史中但沒有輸出，可能失敗了
                    return {
                        'status': 'failed',
                        'progress': 'error',
                        'data': None,
                        'message': 'Image generation failed'
                    }
        
        # 都沒找到，可能還在處理或出錯
        return {
            'status': 'unknown',
            'progress': 'checking',
            'data': None,
            'message': 'Unable to determine status'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'progress': 'error',
            'data': None,
            'message': f'Error checking status: {str(e)}'
        }

def wait_for_image_completion(prompt_id: str, max_wait_time: int = 300, check_interval: int = 10) -> Dict[str, Any]:
    """
    等待圖片生成完成
    
    Args:
        prompt_id (str): ComfyUI返回的prompt ID
        max_wait_time (int): 最大等待時間（秒），默認5分鐘
        check_interval (int): 檢查間隔（秒），默認10秒
        
    Returns:
        Dict[str, Any]: 最終結果
    """
    start_time = time.time()
    last_status = None
    
    print(f"開始等待圖片生成完成 (prompt_id: {prompt_id})")
    
    while time.time() - start_time < max_wait_time:
        status_info = check_image_generation_status(prompt_id)
        current_status = status_info['status']
        
        # 只在狀態改變時印出訊息
        if current_status != last_status:
            print(f"狀態更新: {status_info['message']}")
            last_status = current_status
        
        if current_status == 'completed':
            total_time = int(time.time() - start_time)
            print(f"✅ 圖片生成完成！總耗時: {total_time}秒")
            return status_info
        elif current_status == 'failed':
            print(f"❌ 圖片生成失敗")
            return status_info
        elif current_status == 'error':
            print(f"⚠️ 檢查狀態時出錯: {status_info['message']}")
            return status_info
        
        # 等待下一次檢查
        time.sleep(check_interval)
    
    # 超時
    total_time = int(time.time() - start_time)
    print(f"⏰ 等待超時 ({total_time}秒)")
    return {
        'status': 'timeout',
        'progress': 'timeout',
        'data': None,
        'message': f'Wait timeout after {total_time} seconds'
    }

def get_tags_and_file_info_from_comfy_history(prompt_id: str, max_attempts: int = 10) -> Dict[str, Optional[str]]:
    """
    從ComfyUI歷史記錄中獲取標籤和文件信息（保留原功能以兼容現有代碼）
    
    Args:
        prompt_id (str): The prompt ID returned from ComfyUI
        max_attempts (int): Maximum number of attempts to check for completion
        
    Returns:
        Dict[str, Optional[str]]: Dictionary with 'tags', 'file_name', and 'file_path'
    """
    # 使用新的等待機制
    result_info = wait_for_image_completion(prompt_id, max_wait_time=max_attempts * 20, check_interval=2)
    
    if result_info['status'] == 'completed' and result_info['data']:
        return result_info['data']
    else:
        print(f"Could not retrieve complete info for prompt_id {prompt_id}: {result_info['message']}")
        return {
            'tags': None,
            'file_name': None,
            'file_path': None
        }

def update_task_status_in_database(task_id: str, status: str, result_data: Dict = None):
    """
    更新任務在數據庫中的狀態
    
    Args:
        task_id (str): 任務ID
        status (str): 任務狀態 ('submitted', 'processing', 'completed', 'failed')
        result_data (Dict): 結果數據（當狀態為completed時提供）
    """
    try:
        # 這裡可以根據你的數據庫架構來實現
        # 例如插入到一個任務狀態表中
        
        # 暫時只是打印日誌，實際實現時可以連接到你的PostgreSQL數據庫
        print(f"[DB UPDATE] Task {task_id[:8]}... status: {status}")
        if result_data:
            print(f"[DB UPDATE] Result: {result_data.get('file_name', 'N/A')}")
        
        # 實際數據庫更新代碼示例：
        # from src.database.postgres_handler import update_task_status
        # update_task_status(task_id, status, result_data)
        
    except Exception as e:
        print(f"[DB ERROR] Failed to update task {task_id}: {str(e)}")

def monitor_tasks_to_json(task_infos: List[Dict], output_json_file: str = None):
    """
    背景監控任務並將完成的圖片信息保存到JSON文件
    
    Args:
        task_infos (List[Dict]): 任務信息列表，包含task_id, prompt, scenario等
        output_json_file (str): 輸出JSON文件路徑，如果為None則自動生成
    """
    def monitor_worker():
        if not output_json_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"background_completed_images_{timestamp}.json"
        else:
            json_filename = output_json_file
        
        completed_results = []
        
        print(f"🔄 開始背景監控 {len(task_infos)} 個任務...")
        
        for i, task_info in enumerate(task_infos):
            task_id = task_info["task_id"]
            print(f"⏳ 監控任務 {i+1}/{len(task_infos)} (ID: {task_id[:8]}...)")
            
            try:
                # 等待任務完成，較長的等待時間用於背景處理
                result_info = wait_for_image_completion(task_id, max_wait_time=600, check_interval=20)
                
                if result_info['status'] == 'completed' and result_info['data']:
                    file_info = result_info['data']
                    
                    # 構建完整的結果數據
                    completed_item = {
                        "task_id": task_id,
                        "prompt": task_info.get("prompt", ""),
                        "scenario": task_info.get("scenario", ""),
                        "file_name": file_info.get('file_name'),
                        "file_path": file_info.get('file_path'),
                        "tags": file_info.get('tags'),
                        "status": "completed",
                        "submitted_at": task_info.get("submitted_at"),
                        "completed_at": datetime.now().isoformat()
                    }
                    
                    completed_results.append(completed_item)
                    
                    print(f"✅ 任務 {i+1} 完成")
                    print(f"   檔案: {file_info.get('file_name', 'N/A')}")
                    print(f"   標籤: {file_info.get('tags', 'N/A')[:60]}..." if file_info.get('tags') and len(file_info.get('tags', '')) > 60 else f"   標籤: {file_info.get('tags', 'N/A')}")
                    
                    # 即時保存到JSON（增量更新）
                    try:
                        # 讀取現有數據
                        try:
                            with open(json_filename, 'r', encoding='utf-8') as f:
                                existing_data = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError):
                            existing_data = []
                        
                        # 添加新完成的項目
                        existing_data.append(completed_item)
                        
                        # 保存更新的數據
                        with open(json_filename, 'w', encoding='utf-8') as f:
                            json.dump(existing_data, f, ensure_ascii=False, indent=2)
                        
                        print(f"💾 已保存到 {json_filename}")
                        
                    except Exception as save_error:
                        print(f"❌ 保存JSON時出錯: {save_error}")
                
                else:
                    print(f"❌ 任務 {i+1} 失敗或超時: {result_info.get('message', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ 監控任務 {i+1} 時出錯: {str(e)}")
        
        print(f"🎉 背景監控完成！成功處理 {len(completed_results)} 個任務")
        print(f"📄 結果已保存到: {json_filename}")
        
        return completed_results
    
    # 在背景線程中運行監控
    monitor_thread = threading.Thread(target=monitor_worker, daemon=False)  # 不使用daemon以確保完成
    monitor_thread.start()
    return monitor_thread

def batch_generate_images_async(num_images: int = 10, append_to_file: str = None, theme: str = "fantasy_adventure", wait_for_completion: bool = True) -> Dict[str, Any]:
    """
    異步批量生成圖片，支持後台處理和狀態追蹤
    
    Args:
        num_images (int): 要生成的圖片數量，默認10張
        append_to_file (str): 要附加到的現有JSON文件名，如果為None則創建新文件
        theme (str): 主題類型
        wait_for_completion (bool): 是否等待所有任務完成，默認True
    
    Returns:
        Dict[str, Any]: 包含任務狀態和ID的結果
    """

    if theme == "magical_quest":
        fantasy_scenarios = magical_quest_scenarios
    elif theme == "epic_battle":
        fantasy_scenarios = epic_battle_scenarios
    elif theme == "astrology":
        fantasy_scenarios = astrology_scenarios
    else:
        fantasy_scenarios = fantasy_adventure_scenarios
    
    submitted_tasks = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"開始異步批量生成 {num_images} 張圖片...")
    
    # 階段1: 提交所有任務
    for i in range(num_images):
        try:
            scenario = fantasy_scenarios[i % len(fantasy_scenarios)]
            print(f"\n=== 提交第 {i+1} 張圖片任務 ===")
            print(f"場景描述: {scenario}")
            
            # 生成圖片prompt
            generated_prompt = generate_image_prompt_fun(scenario)
            path_name = f"batch_{theme}"
            
            # 提交任務到ComfyUI
            prompt_id = call_image_request_function(generated_prompt, path_name)
            
            if prompt_id:
                task_info = {
                    "task_id": prompt_id,
                    "prompt": generated_prompt,
                    "scenario": scenario,
                    "path_name": path_name,
                    "submitted_at": datetime.now().isoformat(),
                    "status": "submitted",
                    "index": i + 1
                }
                submitted_tasks.append(task_info)
                # 更新數據庫狀態
                update_task_status_in_database(prompt_id, "submitted")
                print(f"✅ 任務已提交，prompt_id: {prompt_id}")
            else:
                print(f"❌ 第 {i+1} 張圖片任務提交失敗")
                
            # 短暫延遲避免API限制
            time.sleep(0.5)
            
        except Exception as e:
            print(f"提交第 {i+1} 張圖片任務時發生錯誤: {str(e)}")
            continue
    
    print(f"\n✅ 所有任務已提交！成功提交: {len(submitted_tasks)}/{num_images} 個任務")
    
    if not wait_for_completion:
        # 不等待完成，啟動背景JSON監控
        monitor_thread = monitor_tasks_to_json(submitted_tasks, append_to_file)
        
        return {
            "status": "submitted",
            "submitted_count": len(submitted_tasks),
            "total_count": num_images,
            "tasks": submitted_tasks,
            "monitor_thread": monitor_thread,
            "message": "All tasks submitted, background JSON monitoring started"
        }
    
    # 階段2: 等待所有任務完成
    print("\n開始等待所有任務完成...")
    completed_data = []
    
    for task_info in submitted_tasks:
        prompt_id = task_info["task_id"]
        print(f"\n--- 等待任務 {task_info['index']}/{len(submitted_tasks)} 完成 (ID: {prompt_id[:8]}...) ---")
        
        # 等待單個任務完成
        result_info = wait_for_image_completion(prompt_id, max_wait_time=180, check_interval=5)
        
        if result_info['status'] == 'completed' and result_info['data']:
            file_info = result_info['data']
            
            # 構建JSON數據結構
            item_data = {
                "task_id": prompt_id,
                "prompt": task_info["prompt"],
                "scenario": task_info["scenario"],
                "file_name": file_info['file_name'] if file_info['file_name'] else task_info["path_name"],
                "tags": file_info['tags'] if file_info['tags'] else "No tags retrieved",
                "status": "completed",
                "submitted_at": task_info["submitted_at"],
                "completed_at": datetime.now().isoformat()
            }
            
            if file_info['file_path']:
                item_data["file_path"] = file_info['file_path']
            
            completed_data.append(item_data)
            
            print(f"✅ 任務 {task_info['index']} 完成")
            if file_info['tags']:
                print(f"   Tags: {file_info['tags'][:50]}..." if len(file_info['tags']) > 50 else f"   Tags: {file_info['tags']}")
            if file_info['file_name']:
                print(f"   File: {file_info['file_name']}")
        else:
            # 任務失敗或超時
            item_data = {
                "task_id": prompt_id,
                "prompt": task_info["prompt"],
                "scenario": task_info["scenario"],
                "file_name": None,
                "tags": None,
                "status": result_info['status'],
                "error_message": result_info['message'],
                "submitted_at": task_info["submitted_at"],
                "failed_at": datetime.now().isoformat()
            }
            completed_data.append(item_data)
            print(f"❌ 任務 {task_info['index']} 失敗: {result_info['message']}")
    
    # 階段3: 保存結果
    success_count = sum(1 for item in completed_data if item['status'] == 'completed')
    
    theme_key = theme.upper()
    
    if append_to_file:
        output_filename = append_to_file
        try:
            # 讀取現有JSON文件
            try:
                with open(output_filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = {}
            
            # 添加新數據
            if theme_key in existing_data:
                existing_data[theme_key].extend(completed_data)
            else:
                existing_data[theme_key] = completed_data
            
            # 保存更新後的數據
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n🎉 批量生成完成！")
            print(f"成功: {success_count}/{len(submitted_tasks)} 張圖片")
            print(f"數據已附加到: {output_filename}")
            
        except Exception as e:
            print(f"保存數據到JSON文件時發生錯誤: {str(e)}")
    else:
        output_filename = f"batch_image_data_{timestamp}.json"
        try:
            json_data = {theme_key: completed_data}
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n🎉 批量生成完成！")
            print(f"成功: {success_count}/{len(submitted_tasks)} 張圖片")
            print(f"數據已保存到: {output_filename}")
            
        except Exception as e:
            print(f"保存數據到JSON文件時發生錯誤: {str(e)}")
    
    return {
        "status": "completed",
        "submitted_count": len(submitted_tasks),
        "success_count": success_count,
        "total_count": num_images,
        "output_file": output_filename if 'output_filename' in locals() else None,
        "completed_data": completed_data
    }

def start_background_image_generation(num_images: int, theme: str = "astrology", output_json: str = None) -> Dict[str, Any]:
    """
    簡化的背景圖片生成函數 - 提交任務後立即返回，背景監控並保存到JSON
    
    Args:
        num_images (int): 要生成的圖片數量
        theme (str): 主題 ("astrology", "fantasy_adventure", "magical_quest", "epic_battle")
        output_json (str): 輸出JSON文件名，如果為None則自動生成
        
    Returns:
        Dict[str, Any]: 包含提交狀態和監控線程的信息
    """
    print(f"🚀 啟動背景圖片生成: {num_images} 張 {theme} 主題圖片")
    
    result = batch_generate_images_async(
        num_images=num_images,
        theme=theme,
        append_to_file=output_json,
        wait_for_completion=False  # 關鍵：不等待，背景處理
    )
    
    print(f"✅ 任務已提交！{result['submitted_count']} 個任務正在背景處理中")
    if output_json:
        print(f"📄 結果將保存到: {output_json}")
    else:
        print(f"📄 結果將自動保存到時間戳命名的JSON文件")
    
    return result

def check_multiple_task_status(task_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    批量檢查多個任務的狀態
    
    Args:
        task_ids (List[str]): 要檢查的任務ID列表
        
    Returns:
        Dict[str, Dict[str, Any]]: 每個任務ID對應的狀態信息
    """
    results = {}
    for task_id in task_ids:
        results[task_id] = check_image_generation_status(task_id)
    return results

def wait_and_collect_task_results(task_ids: List[str], output_json_file: str = None, max_wait_per_task: int = 600) -> Dict[str, Any]:
    """
    等待指定的task_id列表完成，並收集結果保存到JSON
    
    Args:
        task_ids (List[str]): 要等待的任務ID列表
        output_json_file (str): 輸出JSON文件名，如果為None則自動生成
        max_wait_per_task (int): 每個任務最大等待時間（秒），默認10分鐘
        
    Returns:
        Dict[str, Any]: 包含所有完成任務結果的字典
    """
    if not output_json_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json_file = f"collected_task_results_{timestamp}.json"
    
    print(f"🔄 開始等待並收集 {len(task_ids)} 個任務的結果...")
    
    completed_results = []
    failed_results = []
    
    for i, task_id in enumerate(task_ids):
        print(f"\n⏳ 處理任務 {i+1}/{len(task_ids)} (ID: {task_id[:8]}...)")
        
        try:
            # 等待單個任務完成
            result_info = wait_for_image_completion(task_id, max_wait_time=max_wait_per_task, check_interval=15)
            
            if result_info['status'] == 'completed' and result_info['data']:
                file_info = result_info['data']
                
                # 構建完成的任務結果
                completed_item = {
                    "task_id": task_id,
                    "file_name": file_info.get('file_name'),
                    "file_path": file_info.get('file_path'),
                    "tags": file_info.get('tags'),
                    "status": "completed",
                    "completed_at": datetime.now().isoformat()
                }
                
                completed_results.append(completed_item)
                
                print(f"✅ 任務 {i+1} 完成")
                print(f"   檔案: {file_info.get('file_name', 'N/A')}")
                print(f"   路徑: {file_info.get('file_path', 'N/A')}")
                print(f"   標籤: {file_info.get('tags', 'N/A')[:50]}..." if file_info.get('tags') and len(file_info.get('tags', '')) > 50 else f"   標籤: {file_info.get('tags', 'N/A')}")
                
            else:
                # 任務失敗或超時
                failed_item = {
                    "task_id": task_id,
                    "status": result_info['status'],
                    "error_message": result_info.get('message', 'Unknown error'),
                    "failed_at": datetime.now().isoformat()
                }
                
                failed_results.append(failed_item)
                print(f"❌ 任務 {i+1} 失敗: {result_info.get('message', 'Unknown error')}")
                
        except Exception as e:
            failed_item = {
                "task_id": task_id,
                "status": "error",
                "error_message": str(e),
                "failed_at": datetime.now().isoformat()
            }
            failed_results.append(failed_item)
            print(f"❌ 任務 {i+1} 處理時出錯: {str(e)}")
    
    # 提取image_paths列表
    image_paths = [item.get('file_path') for item in completed_results if item.get('file_path')]
    
    # 構建最終結果
    final_result = {
        "image_paths": image_paths,
        "completed_tasks": completed_results,
        "failed_tasks": failed_results,
        "generated_at": datetime.now().isoformat()
    }
    
    # 保存到JSON文件
    try:
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 處理完成！")
        print(f"✅ 成功: {len(completed_results)}/{len(task_ids)} 個任務")
        print(f"❌ 失敗: {len(failed_results)} 個任務")
        print(f"📄 結果已保存到: {output_json_file}")
        print(f"🎯 圖片路徑: {len(image_paths)} 個文件路徑已提取")
        
    except Exception as save_error:
        print(f"❌ 保存JSON文件時出錯: {save_error}")
    
    return final_result


if __name__ == "__main__":
    # 使用示例1: 傳統方式 - 等待所有任務完成
    # result = batch_generate_images_async(5, theme="astrology", wait_for_completion=True)
    
    # 主要使用方式：背景生成並保存到JSON
    # result = start_background_image_generation(
    #     num_images=5,
    #     theme="astrology",
    #     output_json="my_generated_images.json"  # 可選，不指定則自動命名
    # )
    
    # print(f"🔄 背景處理已啟動，程序可以繼續執行其他任務...")
    # print(f"監控線程正在處理 {result['submitted_count']} 個任務")
    task_ids = ['1309572b-a327-4a95-b9b7-900134a2a166', 'fa5630f7-7721-452c-be5f-a41f80329ddf', '76acc661-eaf8-4a25-b18f-5fb389becb84', 'a3b13cf2-9da1-4a70-8866-4f2804f36df9', '75b56851-354e-43c1-8259-dfbd54dd16fa', '4b5b5a57-3404-4161-b69e-dfbca3f79c18', '210294c1-3e33-46c2-8829-f422ef27975b', '0f92aa5e-695d-4cb0-9b74-0c11247ccece', '0f0b9825-3575-4053-b01d-a76f2e563690', '2f1ec3d3-b2e4-4921-8118-5db7a5954732']    
    wait_and_collect_task_results(task_ids=task_ids, output_json_file="my_generated_images2.json")    

    # 你的程序可以繼續執行其他工作
    # 圖片會在背景中生成完成，並自動保存詳細信息到JSON文件
    
    # 如果需要檢查進度（可選）
    # import time
    # time.sleep(60)  # 等待1分鐘
    # task_ids = [task["task_id"] for task in result["tasks"]]
    # status_check = check_multiple_task_status(task_ids)
    # print("當前狀態:", status_check)