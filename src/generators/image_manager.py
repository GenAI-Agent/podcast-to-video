#!/usr/bin/env python3
"""
Image Manager
Handles image selection, art style selection, image script generation, and image path management
"""
import os
import json
from typing import List, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import sys

# Add project root to sys.path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)
from src.generators.prompts.image_prompt import ART_STYLE_PROMPT, IMAGE_PROMPT, IMAGE_USER_PROMPT
from src.database.pinecone_handler import PineconeHandler
from src.generators.image_generator import call_image_request_function, generate_image_prompt_fun
# Load environment variables
load_dotenv()


class ImageManager:
    """Manager class for handling image-related operations"""
    
    def __init__(self):
        """Initialize the image manager
        
        Args:
            pinecone_handler: Optional Pinecone handler for vector search
        """
        self.pinecone_handler = PineconeHandler()
    
    def select_art_style(self, transcript: str) -> str:
        """Select art style based on transcript content using GPT
        
        Args:
            transcript: The transcript content
            
        Returns:
            str: Selected art style name
        """
        try:
            # Get Azure OpenAI credentials from environment
            api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")

            if not api_base or not api_key:
                print("Warning: Azure OpenAI credentials not found. Using default art style.")
                return "Digital Painting (Rich Colour)"

            # Initialize Azure OpenAI
            llm = AzureChatOpenAI(
                azure_endpoint=api_base,
                api_key=api_key,
                azure_deployment="gpt-4o-testing",
                api_version="2025-01-01-preview",
                temperature=0.3,  # Lower temperature for more consistent style selection
                max_tokens=50,
                timeout=None,
                max_retries=2,
            )

            # Create prompt template for art style selection
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", ART_STYLE_PROMPT),
                ("human", "{transcript}"),
            ])

            # Generate art style selection
            chain = prompt_template | llm | StrOutputParser()
            art_style = chain.invoke({"transcript": transcript}).strip()
            
            print(f"✓ Selected art style: {art_style}")
            return art_style

        except Exception as e:
            print(f"Error selecting art style: {e}")
            return "Digital Painting (Rich Colour)"  # Default fallback

    def get_image_list_script(self, transcript: str, audio_duration: float, art_style: str) -> List[dict]:
        """Generate a list of image descriptions with durations for video creation using GPT

        Args:
            transcript: The transcript content
            audio_duration: Total duration of the audio in seconds

        Returns:
            List[dict]: A list of dictionaries with format [{description:"tags...",duration:1.2},...]
        """
        try:
            # Get Azure OpenAI credentials from environment
            api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")

            if not api_base or not api_key:
                print(
                    "Warning: Azure OpenAI credentials not found. Returning empty list."
                )
                return []

            # Initialize Azure OpenAI
            llm = AzureChatOpenAI(
                azure_endpoint=api_base,
                api_key=api_key,
                azure_deployment="gpt-4o-testing",
                api_version="2025-01-01-preview",
                temperature=0.7,
                max_tokens=2000,
                timeout=None,
                max_retries=2,
            )

            # Create prompt template for image script generation
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", IMAGE_PROMPT),
                    ("human", IMAGE_USER_PROMPT),
                ]
            )

            # Generate image script
            chain = prompt_template | llm | StrOutputParser()
            response = chain.invoke(
                {"transcript": transcript, "audio_duration": audio_duration, "art_style": art_style}
            )

            # Parse the JSON response
            try:
                # Extract JSON array from the response
                # Sometimes GPT might add extra text, so we need to find the JSON part
                json_start = response.find("[")
                json_end = response.rfind("]") + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    image_list = json.loads(json_str)
                else:
                    raise ValueError("No JSON array found in response")

                # Validate and ensure minimum duration
                validated_list = []
                for item in image_list:
                    if (
                        isinstance(item, dict)
                        and "prompt" in item
                        and "duration" in item
                    ):
                        # Ensure minimum duration of 1 second
                        duration = max(1.0, float(item["duration"]))
                        validated_list.append(
                            {
                                "prompt": str(item["prompt"]),
                                "duration": duration,
                            }
                        )

                # Adjust durations to match audio duration exactly
                total_duration = sum(item["duration"] for item in validated_list)
                if total_duration > 0 and abs(total_duration - audio_duration) > 0.1:
                    # Scale durations proportionally
                    scale_factor = audio_duration / total_duration
                    for item in validated_list:
                        item["duration"] = max(1.0, item["duration"] * scale_factor)

                print(
                    f"✓ Successfully generated {len(validated_list)} image prompts with durations"
                )
                return validated_list

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Response was: {response}")
                return []

        except Exception as e:
            print(f"Error generating image script: {e}")
            return []

    def vector_search_images(self, prompts: List[str], top_k: int = 5) -> List[str]:
        """Search for images using vector similarity for each sentence with deduplication"""
        image_paths = []
        used_images = set()  # Track used images to avoid duplicates

        if not self.pinecone_handler:
            print("Warning: Pinecone handler not initialized, using dummy image paths")
            return [None] * len(prompts)

        for prompt in prompts:
            try:
                # Use PineconeHandler to search with higher top_k for more options
                results = self.pinecone_handler.query_pinecone(
                    query=prompt,
                    metadata_filter={},  # No filter for now
                    index_name="image-library",
                    namespace="description",
                    top_k=top_k,  # Get more candidates
                )

                selected_image = None

                if results and len(results) > 0:
                    # Try to find an unused image from the results
                    for result in results:
                        if "metadata" in result and "file_path" in result["metadata"]:
                            candidate_path = result["metadata"]["file_path"]

                            # Check if this image hasn't been used yet
                            if candidate_path not in used_images:
                                selected_image = candidate_path
                                used_images.add(candidate_path)
                                print(
                                    f"Found unique image for '{prompt[:30]}...': {candidate_path}"
                                )
                                break

                    # If all top results are already used, use the best match anyway
                    if selected_image is None and results:
                        fallback_result = results[0]
                        if (
                            "metadata" in fallback_result
                            and "file_path" in fallback_result["metadata"]
                        ):
                            selected_image = fallback_result["metadata"]["file_path"]
                            print(
                                f"Using duplicate image (no unique found) for '{prompt[:30]}...': {selected_image}"
                            )

                if selected_image:
                    image_paths.append(selected_image)
                else:
                    print(
                        f"Warning: No valid images found for description: {prompt[:50]}..."
                    )
                    image_paths.append(None)

            except Exception as e:
                print(f"Error searching for description '{prompt[:50]}...': {e}")
                image_paths.append(None)

        print(
            f"📊 Image diversity: {len(used_images)} unique images selected for {len(prompts)} prompts"
        )
        return image_paths

    def convert_windows_path_to_wsl(self, windows_path: str) -> str:
        """Convert Windows path to WSL path"""
        if windows_path and windows_path.startswith("C:"):
            # Convert C:\path\to\file to /mnt/c/path/to/file
            wsl_path = windows_path.replace("C:", "/mnt/c").replace("\\", "/")
            return wsl_path
        return windows_path

    def find_fallback_image(self, original_path: str, used_fallbacks: set = None) -> Optional[str]:
        """Find a fallback image in the same directory when the original is not found"""
        if used_fallbacks is None:
            used_fallbacks = set()

        wsl_path = self.convert_windows_path_to_wsl(original_path)
        directory = os.path.dirname(wsl_path)

        if os.path.exists(directory):
            # Look for .png files in the same directory, avoiding already used fallbacks
            available_files = []
            for file in os.listdir(directory):
                if file.endswith(".png"):
                    fallback_path = os.path.join(directory, file)
                    if (
                        os.path.exists(fallback_path)
                        and fallback_path not in used_fallbacks
                    ):
                        available_files.append(fallback_path)

            # If we found unused files, return the first one
            if available_files:
                selected_fallback = available_files[0]
                used_fallbacks.add(selected_fallback)
                print(f"🔄 Using unique fallback image: {selected_fallback}")
                return selected_fallback

            # If all files are used, just pick the first available file
            for file in os.listdir(directory):
                if file.endswith(".png"):
                    fallback_path = os.path.join(directory, file)
                    if os.path.exists(fallback_path):
                        print(
                            f"🔄 Using fallback image (no unique available): {fallback_path}"
                        )
                        return fallback_path
        return None

    def batch_call_image_generator(self, list_image_script: List[dict]) -> List[str]:
        """Call image generator"""
        task_id_list = []
        for item in list_image_script:
            prompt = item['prompt']
            optimized_prompt = generate_image_prompt_fun(prompt)
            prompt_id = call_image_request_function(optimized_prompt, "test_prompt")
            task_id_list.append(prompt_id)
        return task_id_list

if __name__ == "__main__":
    image_manager = ImageManager()
    # text="大家好，今天要和大家分享一本令人深思的書——侯文詠的《變成自己想望的大人》。這是他成長四部曲的最終篇，凝聚了他一路走來的生命故事。侯文詠用他的文字告訴我們，人生的答案不是標準化的，而是從每個人的獨特經歷中尋找。從醫師到作家，他勇敢辭職，踏上未知的道路，面對挫折與傷痛，卻在其中找到韌性與熱情。這本書寫給那些敢於「好玩」、「不乖」的人，它提醒我們，每一天都要朝心中的那個理想自己邁進。或許有一天，我們也能成為迷霧中的光，陪伴他人走過黑夜。"
    # transcript="大家好，今天要和大家分享侯文詠的新書《變成自己想望的大人》。這是他成長四部曲的最終篇，凝聚了他一路走 來的生命故事。從不被看好的寫作天賦，到成為醫師、作家，再到編劇與主持人，侯文詠始終跟隨內心的喜歡與熱情，繞過挫折和風雨，收穫了出乎意料的風景。他說，人生沒有標準答案，所有的功課都要自己去面對。這本書寫 給那些走在「好玩」、「不乖」道路上的人，帶著生命的光，去成為自己想望的大人。希望大家能從這本書中，找到屬於自己的答案！"
    # art_style = image_manager.select_art_style(transcript)
    # list_image_script = image_manager.get_image_list_script(transcript, 60, art_style)
    # print(list_image_script)
    image_script = [{'prompt': 'A flat vector design illustration of a person sitting at a desk with coins and a piggy bank, looking confident and motivated to save money.', 'duration': 4.504907700649906}, {'prompt': "A flat vector design of a book cover titled '上癮式存錢' with vibrant colors and an image of a growing money tree.", 'duration': 3.6039261605199253}, {'prompt': 'A flat vector design showing a person balancing work and saving money, with a clock and money stacks symbolizing limited work hours and wealth accumulation.', 'duration': 4.504907700649906}, {'prompt': 'A flat vector design of investment tools such as graphs, charts, and coins growing steadily while a person observes with satisfaction.', 'duration': 3.6039261605199253}, {'prompt': "A flat vector design of a person standing confidently amidst a storm symbolizing a 'black swan' event, with their financial assets staying stable.", 'duration': 3.6039261605199253}, {'prompt': 'A flat vector design showing the evolution of financial habits from saving small amounts to making wise spending choices.', 'duration': 4.504907700649906}, {'prompt': 'A flat vector design of a person managing multiple income streams, including a main job and side hustles, with money flowing into a savings account.', 'duration': 3.6039261605199253}, {'prompt': 'A flat vector design illustrating the concept of compound interest as a growing snowball rolling downhill, accumulating more wealth.', 'duration': 3.6039261605199253}, {'prompt': 'A flat vector design highlighting the power of a single coin, with a coin transforming into a financial accelerator through action and determination.', 'duration': 3.0601909516986283}, {'prompt': "A flat vector design of a promotional banner with '75折' written prominently, showcasing the book as a stepping stone to financial freedom.", 'duration': 1.8019630802599627}]
    task_id_list = image_manager.batch_call_image_generator(image_script)
    print(task_id_list)