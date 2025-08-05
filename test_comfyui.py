import json
import requests
import random
import time
from typing import Dict, Optional

def test_comfyui_connection():
    """Test ComfyUI connection with a simple fantasy prompt"""
    
    # Test fantasy prompt
    test_prompt = "Ancient dragon's lair with treasure hoard and magical artifacts, cinematic lighting, professional photography"
    
    try:
        # ComfyUI API endpoint
        COMFY_URL = "https://6e0bf634b876.ngrok-free.app/api/prompt"
        
        # Load workflow with UTF-8 encoding
        workflow_path = "workflows/TaggedExport.json"
        with open(workflow_path, "r", encoding='utf-8') as file:
            workflow = json.load(file)

        # Process workflow nodes to replace placeholders
        for node in workflow.values():
            # Replace prompt text in CLIPTextEncode nodes
            if node.get("class_type") == "CLIPTextEncode":
                if "inputs" in node and "text" in node["inputs"]:
                    node["inputs"]["text"] = test_prompt
            
            # Update SaveImageExtended to use custom folder name
            if node.get("class_type") == "SaveImageExtended":
                if "inputs" in node:
                    if "foldername_keys" in node["inputs"]:
                        node["inputs"]["foldername_keys"] = "test_fantasy"
                    if "filename_prefix" in node["inputs"]:
                        node["inputs"]["filename_prefix"] = "test_fantasy_%F_%H-%M-%S"
                    if "filename_keys" in node["inputs"]:
                        node["inputs"]["filename_keys"] = ""
        
        print(f"Testing ComfyUI connection...")
        print(f"Prompt: {test_prompt}")
        print(f"ComfyUI URL: {COMFY_URL}")
        
        # Send request to ComfyUI
        response = requests.post(COMFY_URL, json={"prompt": workflow})
        
        if response.status_code == 200:
            prompt_id = response.json().get('prompt_id')
            print(f"✅ ComfyUI connection successful!")
            print(f"Prompt ID: {prompt_id}")
            return prompt_id
        else:
            print(f"❌ ComfyUI request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing ComfyUI: {str(e)}")
        return None

if __name__ == "__main__":
    test_comfyui_connection() 