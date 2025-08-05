# -*- coding: utf-8 -*-
# usage: quantitativeAgent.py LensGraph.py LawGraph.py

import sys
import os
import asyncio
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import json
import psycopg2

from dotenv import load_dotenv
from openai import AzureOpenAI
from openai import OpenAI as OpenAIClient
from pinecone import Pinecone 
import uuid
load_dotenv()

class PineconeHandler:
    """
    封裝 Pinecone 相關操作以及 Azure OpenAI 向量生成的類別
    """

    def __init__(self):
        # 從環境變數讀取 API Key
        self._pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self._azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self._azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self._nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        self._embed_client = AzureOpenAI(
            api_key=self._azure_openai_api_key,
            api_version="2024-08-01-preview",
            azure_endpoint=self._azure_openai_endpoint,
        )
        # 初始化 NVIDIA API 客戶端
        self._pc = Pinecone(api_key=self._pinecone_api_key)
        
        # 資料庫連接字串
        self._database_url = "postgresql://postgres:1234@ec2-52-194-194-37.ap-northeast-1.compute.amazonaws.com:5434/image"

    def embedder(self, query: str) -> List[float]:
        """
        調用openAI的embedder，將query轉換成向量
        Args:
            query (str): 用戶query
        Returns:
            List[float]: 向量
        """
        response = self._embed_client.embeddings.create(
            model="text-embedding-3-small", 
            input=query,
            dimensions=512  # 設定為 512 維度以匹配 Pinecone 索引
        )
        return response.data[0].embedding
    
    def check_existing_ids(self, index_name: str, namespace: str, ids: List[str]) -> set:
        """檢查向量ID是否已存在於Pinecone中"""
        try:
            index = self._pc.Index(index_name, pool_threads=50)
            existing_vectors = index.fetch(ids=ids, namespace=namespace)
            return set(existing_vectors.vectors.keys())
        except Exception as e:
            print(f"檢查ID時發生錯誤: {str(e)}")
            return set()

    def upsert_pinecone(self, index_name: str, namespace: str, embeddedData: List[Dict]) -> None:
        """
        批量上傳向量到Pinecone
        Args:
            index_name (str): Pinecone索引名稱
            namespace (str): Pinecone命名空間
            embeddedData (List[Dict]): 要上傳的數據列表，每個字典包含id、metadata和value
        """
        index = self._pc.Index(index_name, pool_threads=50)
        
        if not embeddedData:
            return
            
        vectors = []
        for data in embeddedData:
            vector = self.embedder(data["value"]) if isinstance(data.get("value"), str) else data.get("values", [])
            vectors.append({
                "id": data["id"],
                "values": vector,
                "metadata": data["metadata"]
            })
        if vectors:
            try:
                index.upsert(vectors=vectors, namespace=namespace)
            except Exception as e:
                print(f"批量上傳向量時發生錯誤: {str(e)}")

    def search_cache(self, userQuery: str,metadata_filter: dict, namespace: str):
        try:
            matcheData = self.query_pinecone(query=userQuery, metadata_filter=metadata_filter, index_name='cache', namespace=namespace, top_k=1)
            if matcheData and matcheData[0]["score"] > 0.9:
                # 處理sources字段 - 可能是JSON字符串，需要解析
                sources = matcheData[0]["metadata"].get("sources", "")
                if sources and isinstance(sources, str):
                    try:
                        sources = json.loads(sources)
                    except:
                        sources = []
                
                return {
                    "answer": matcheData[0]["metadata"].get("answer", ""), 
                    "prompts": matcheData[0]["metadata"].get("prompts", []), 
                    "sources": sources
                }
            else:
                return None
        except Exception as e:
            print(f"Error in search_cache: {str(e)}")
            return None

    def query_pinecone(self, query: str, metadata_filter: dict, index_name: str = '', namespace: str = '', top_k: int = 5) -> List[Dict]:
        """
        調用Pinecone，並使用結構化檢索+向量檢索書籍
        Args:
            query (str): 用戶詢問詞
            metadata_filter (dict): metadata過濾條件
            index_name (str): 索引名稱
            namespace (str): 命名空間
            top_k (int): 返回結果數量
        Returns:
            List[Dict]: 搜索結果，作為給用戶的回答
        """
        index = self._pc.Index(index_name, pool_threads=50)
        if query:
            embed_response = self._embed_client.embeddings.create(
                model="text-embedding-3-small", 
                input=query,
                dimensions=512
            )
            vector = embed_response.data[0].embedding
        else:
            vector = [0] * 512
            
        results = index.query(
            namespace=namespace,
            vector=vector,
            top_k=top_k,
            filter=metadata_filter,
            include_values=False,
            include_metadata=True,
        )
        
        return results["matches"]

    async def query_pinecone_async(self, query: str, metadata_filter: dict, index_name: str = '', namespace: str = '', top_k: int = 5) -> List[Dict]:
        """
        調用Pinecone，並使用結構化檢索+向量檢索書籍 (非同步)
        Args:
            query (str): 用戶詢問詞
            metadata_filter (dict): metadata過濾條件
            index_name (str): 索引名稱
            namespace (str): 命名空間
            top_k (int): 返回結果數量
        Returns:
            List[Dict]: 搜索結果，作為給用戶的回答
        """
        index = self._pc.Index(index_name, pool_threads=50)
        
        if query:
            embed_response = self._embed_client.embeddings.create(
                model="text-embedding-3-small", 
                input=query,
                dimensions=512
            )
            vector = embed_response.data[0].embedding
        else:
            vector = [0] * 512
        
        results = index.query(
            namespace=namespace,
            vector=vector,
            top_k=top_k,
            filter=metadata_filter,
            include_values=False,
            include_metadata=True,
        )
        for result in results["matches"]:
            result["metadata"]["keywords"] = [query]

        return results["matches"]

    def query_by_id(self, id: str, index_name: str, namespace: str) -> Dict:
        """
        通過ID直接獲取向量
        Args:
            id (str): 要查詢的ID
            index_name (str): 索引名稱
            namespace (str): 命名空間
        Returns:
            Dict: 包含查詢結果的字典
        """
        index = self._pc.Index(index_name, pool_threads=50)
        results = index.query(
            namespace=namespace,
            id=id,
            top_k=1,
            include_values=False,
            include_metadata=True
        )
        return results['matches'][0]

    async def run_parallel_queries(self, queries: List[str], metadata_filter: dict, index_name: str, namespace: str, top_k: int) -> List[Dict]:
        """
        並行執行多個查詢
        """
        MAX_CONCURRENT_QUERIES = 4
        index = self._pc.Index(index_name)
        
        # 1. 首先批量獲取 embeddings
        loop = asyncio.get_running_loop()
        embed_response = await loop.run_in_executor(
            None,
            lambda: self._embed_client.embeddings.create(
                model="text-embedding-3-small",
                input=queries,
                dimensions=512
            )
        )
        vectors = [data.embedding for data in embed_response.data]
        
        # 2. 定義單個查詢函數
        def run_single_query(query_data):
            vector, original_query = query_data
            result = index.query(
                vector=vector,
                namespace=namespace,
                top_k=top_k,
                filter=metadata_filter,
                include_metadata=True
            )
            # 添加原始查詢詞到結果中
            for match in result.matches:
                match.metadata["keywords"] = [original_query]
            return result.matches
        
        # 3. 並行執行查詢
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_QUERIES) as executor:
            results = list(executor.map(run_single_query, zip(vectors, queries)))
        
        return results

    def update_metadata_batch(self, index_name: str, namespace: str, updates: List[Dict[str, dict]]) -> None:
        """
        批量更新Pinecone中向量的metadata
        Args:
            index_name (str): Pinecone索引名稱
            namespace (str): Pinecone命名空間
            updates (List[Dict[str, dict]]): 要更新的數據列表
        """
        try:
            index = self._pc.Index(index_name, pool_threads=50)
            # 將更新列表分批處理，每批100個
            batch_size = 100
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i + batch_size]
                vectors = [{
                    "id": item["id"],
                    "metadata": item["metadata"],
                    "values": item["values"]
                } for item in batch]
                # 執行批量更新
                index.upsert(vectors=vectors, namespace=namespace)
                
            print(f"成功更新 {len(updates)} 條記錄的metadata")
            
        except Exception as e:
            raise e

    def extract_prompt_from_json(self, prompt_value: str) -> str:
        """
        從 prompt column 中提取實際的 prompt 內容
        Args:
            prompt_value (str): prompt column 的原始值
        Returns:
            str: 提取的 prompt 文本
        """
        try:
            # 首先嘗試解析為 JSON
            data = json.loads(prompt_value.strip())
            
            # 如果有 prompt key，處理其值
            if isinstance(data, dict) and "prompt" in data:
                prompt_content = data["prompt"]
                
                # 如果 prompt 是嵌套的 dict
                if isinstance(prompt_content, dict):
                    # 嘗試多種方式提取文本
                    # 1. 如果有 description 且是字符串
                    if "description" in prompt_content and isinstance(prompt_content["description"], str):
                        return prompt_content["description"]
                    
                    # 2. 如果 description 也是 dict，轉換為 JSON 字符串
                    if "description" in prompt_content and isinstance(prompt_content["description"], dict):
                        return json.dumps(prompt_content["description"], separators=(',', ':'))
                    
                    # 3. 如果沒有 description，將整個 prompt dict 轉為字符串
                    return json.dumps(prompt_content, separators=(',', ':'))
                
                # 如果 prompt 是字符串，直接返回
                elif isinstance(prompt_content, str):
                    return prompt_content
                
                # 其他類型，轉為字符串
                else:
                    return str(prompt_content)
            
            # 如果沒有 prompt key，返回整個 JSON 的字符串表示
            return json.dumps(data, separators=(',', ':')) if isinstance(data, dict) else str(data)
            
        except json.JSONDecodeError:
            # 如果不是有效的 JSON，直接返回原始值
            return prompt_value
        except Exception as e:
            print(f"提取 prompt 時發生錯誤: {str(e)}")
            return prompt_value

    def fetch_image_library_data(self, limit: int = None, offset: int = 0) -> List[Dict]:
        """
        從 image_library 表中獲取資料
        Args:
            limit (int): 限制獲取的記錄數量
            offset (int): 偏移量
        Returns:
            List[Dict]: 資料庫記錄列表
        """
        try:
            conn = psycopg2.connect(self._database_url)
            cursor = conn.cursor()
            
            query = "SELECT id, name, file_path, prompt, description, theme, sub_theme FROM image_library WHERE status = 'active'"
            if limit:
                query += f" LIMIT {limit} OFFSET {offset}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # 轉換為字典格式
            columns = ['id', 'name', 'file_path', 'prompt', 'description', 'theme', 'sub_theme']
            data = []
            for row in rows:
                record = dict(zip(columns, row))
                data.append(record)
            
            cursor.close()
            conn.close()
            return data
            
        except Exception as e:
            print(f"獲取 image_library 資料時發生錯誤: {str(e)}")
            return []

    def batch_upload_image_library_to_pinecone(self, index_name: str, namespace: str = None, 
                                              batch_size: int = 50, limit: int = None) -> Dict:
        """
        批量將 image_library 資料上傳到 Pinecone
        現在會上傳到兩個 namespace：
        1. namespace='prompt' - 使用 prompt 作為向量
        2. namespace='description' - 使用 description 作為向量
        
        Args:
            index_name (str): Pinecone 索引名稱
            namespace (str): Pinecone 命名空間 (如果指定，將只上傳到該 namespace)
            batch_size (int): 每批處理的記錄數量
            limit (int): 總共處理的記錄數量限制
        Returns:
            Dict: 處理結果統計
        """
        print(f"開始批量上傳 image_library 資料到 Pinecone...")
        print(f"索引: {index_name}")
        
        # 決定要上傳的 namespaces
        if namespace:
            namespaces = [namespace]
            print(f"指定命名空間: {namespace}")
        else:
            namespaces = ['prompt', 'description']
            print(f"將上傳到兩個命名空間: prompt 和 description")
        
        # 統計資訊
        total_stats = {
            "total_processed": 0,
            "successful_uploads": 0,
            "failed_uploads": 0,
            "skipped_existing": 0
        }
        
        try:
            # 獲取資料
            data = self.fetch_image_library_data(limit=limit)
            if not data:
                print("沒有找到資料")
                return total_stats
            
            print(f"共獲取到 {len(data)} 筆記錄")
            
            # 對每個 namespace 進行上傳
            for current_namespace in namespaces:
                print(f"\n{'='*80}")
                print(f"開始上傳到 namespace: {current_namespace}")
                print(f"{'='*80}")
                
                namespace_stats = {
                    "successful_uploads": 0,
                    "failed_uploads": 0
                }
                
                # 分批處理
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    batch_vectors = []
                    
                    print(f"\n處理第 {i//batch_size + 1} 批 ({len(batch)} 筆記錄)...")
                    
                    for record in batch:
                        try:
                            # 根據 namespace 決定使用哪個欄位作為向量
                            if current_namespace == 'prompt':
                                # 提取 prompt 內容
                                vector_text = self.extract_prompt_from_json(record['prompt'])
                                print(f"ID: {record['id'][:8]}... 使用 prompt 作為向量")
                            elif current_namespace == 'description':
                                # 使用 description (tags) 作為向量
                                vector_text = record['description'] or ""
                                if not vector_text:
                                    print(f"ID: {record['id'][:8]}... 沒有 description，跳過")
                                    continue
                                print(f"ID: {record['id'][:8]}... 使用 description 作為向量")
                            
                            # 準備 metadata (兩個 namespace 使用相同的 metadata)
                            metadata = {
                                "name": record['name'],
                                "file_path": record['file_path'],
                                "description": record['description'] or "",
                                "theme": record['theme'],
                                "sub_theme": record['sub_theme'],
                                "original_prompt": record['prompt'][:500] if record['prompt'] else ""
                            }
                            
                            # 準備向量資料
                            vector_data = {
                                "id": record['id'],
                                "value": vector_text,
                                "metadata": metadata
                            }
                            
                            batch_vectors.append(vector_data)
                            
                        except Exception as e:
                            print(f"處理記錄 {record['id']} 時發生錯誤: {str(e)}")
                            namespace_stats["failed_uploads"] += 1
                            continue
                    
                    # 上傳這批資料到 Pinecone
                    if batch_vectors:
                        try:
                            self.upsert_pinecone(index_name, current_namespace, batch_vectors)
                            namespace_stats["successful_uploads"] += len(batch_vectors)
                            print(f"成功上傳 {len(batch_vectors)} 筆記錄到 {current_namespace}")
                        except Exception as e:
                            print(f"上傳批次資料時發生錯誤: {str(e)}")
                            namespace_stats["failed_uploads"] += len(batch_vectors)
                
                # 更新總統計
                total_stats["successful_uploads"] += namespace_stats["successful_uploads"]
                total_stats["failed_uploads"] += namespace_stats["failed_uploads"]
                
                # 顯示此 namespace 的統計
                print(f"\n{current_namespace} namespace 上傳完成：")
                print(f"  成功上傳: {namespace_stats['successful_uploads']}")
                print(f"  失敗上傳: {namespace_stats['failed_uploads']}")
            
            # 計算總處理記錄數
            total_stats["total_processed"] = len(data) * len(namespaces)
            
            # 輸出總統計結果
            print(f"\n{'='*80}")
            print(f"批量上傳完成！")
            print(f"總處理記錄數: {total_stats['total_processed']}")
            print(f"成功上傳: {total_stats['successful_uploads']}")
            print(f"失敗上傳: {total_stats['failed_uploads']}")
            
            return total_stats
            
        except Exception as e:
            print(f"批量上傳過程中發生錯誤: {str(e)}")
            return total_stats

if __name__ == "__main__":
    # 簡易示範如何測試此類別
    pinecone_query = PineconeHandler()
    # 測試批量上傳功能 - 先上傳 10 筆測試
    result = pinecone_query.batch_upload_image_library_to_pinecone(
        index_name="image-library", 
        namespace="test", 
        batch_size=10, 
        limit=10
    )
    print(f"測試結果: {result}")