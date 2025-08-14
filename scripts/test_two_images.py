#!/usr/bin/env python3
import os
import json
import time
import requests
from urllib.parse import urlsplit


COMFY_URL = os.environ.get("COMFY_URL", "https://6e0bf634b876.ngrok-free.app/api/prompt")
WORKFLOW_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workflows", "KreaGen.json")


def run_one(prompt_text: str, path_name: str):
    # Load workflow
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # Inject prompt and ensure save path
    for node in workflow.values():
        if node.get("class_type") == "CLIPTextEncode":
            if "inputs" in node and "text" in node["inputs"]:
                node["inputs"]["text"] = prompt_text

        if node.get("class_type") == "SaveImageExtended":
            if "inputs" in node:
                if "foldername_keys" in node["inputs"]:
                    node["inputs"]["foldername_keys"] = path_name
                if "filename_prefix" in node["inputs"]:
                    node["inputs"]["filename_prefix"] = f"{path_name}_%F_%H-%M-%S"
                if "filename_keys" in node["inputs"]:
                    node["inputs"]["filename_keys"] = ""

        if node.get("class_type") == "SaveImage":
            if "inputs" in node:
                if "subfolder" in node["inputs"]:
                    node["inputs"]["subfolder"] = path_name
                if "filename_prefix" in node["inputs"]:
                    node["inputs"]["filename_prefix"] = f"{path_name}/{path_name}_%F_%H-%M-%S"

    # Submit job
    response = requests.post(COMFY_URL, json={"prompt": workflow}, timeout=120)
    if response.status_code != 200:
        print(f"ComfyUI request failed: {response.status_code} {response.text}")
        return None

    prompt_id = response.json().get("prompt_id")
    if not prompt_id:
        print("No prompt_id returned")
        return None

    # Poll history for result
    parts = urlsplit(COMFY_URL)
    base_url = f"{parts.scheme}://{parts.netloc}"
    history_url = f"{base_url}/history/{prompt_id}"

    for attempt in range(20):  # ~40s
        try:
            history_response = requests.get(history_url, timeout=30)
            if history_response.status_code == 200:
                history_data = history_response.json()
                outputs = history_data.get(prompt_id, {}).get("outputs", {})

                for _node_id, output in outputs.items():
                    if "images" in output and output["images"]:
                        image_info = output["images"][0]
                        filename = image_info.get("filename", "")
                        subfolder = image_info.get("subfolder", "")

                        file_name = os.path.splitext(filename)[0] if filename else None
                        if filename:
                            if subfolder:
                                file_path = f"C:\\Users\\x7048\\Documents\\ComfyUI\\output\\{subfolder}\\{filename}"
                            else:
                                file_path = f"C:\\Users\\x7048\\Documents\\ComfyUI\\output\\{filename}"
                        else:
                            file_path = None

                        return {
                            "task_id": prompt_id,
                            "prompt": prompt_text,
                            "file_name": file_name,
                            "file_path": file_path,
                            "tags": "No tags retrieved",
                        }

            time.sleep(2)
        except Exception as e:
            print(f"Error polling history: {e}")
            time.sleep(2)

    return {
        "task_id": prompt_id,
        "prompt": prompt_text,
        "file_name": None,
        "file_path": None,
        "tags": "No tags retrieved",
    }


def main():
    prompts = [
        "A cinematic fantasy landscape, high detail, 1024x1024",
        "Futuristic city skyline at golden hour, cinematic, 1024x1024",
    ]

    results = []
    for i, p in enumerate(prompts):
        path = f"test_kreagen_{i+1}"
        print(f"Submitting {i+1}: {path}")
        result = run_one(p, path)
        print(f"Result: {result}")
        results.append(result)

    print("\nSummary:")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

