import json
import os
import time
from typing import List, Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GPTTagGenerator:
    def __init__(self):
        """Initialize the GPT tag generator with Azure OpenAI"""
        # Get Azure OpenAI credentials from environment
        self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        if not self.api_base or not self.api_key:
            raise ValueError("Azure OpenAI credentials not found. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables.")
        
        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.api_base,
            api_key=self.api_key,
            azure_deployment="gpt-4o-testing",
            api_version="2025-01-01-preview",
            temperature=0.3,
            max_tokens=100,
            timeout=None,
            max_retries=2,
        )
    
    def generate_tags_for_prompt(self, prompt: str, max_tags: int = 8) -> List[str]:
        """Generate relevant tags for a given prompt using Azure OpenAI"""
        try:
            system_prompt = f"""You are a professional image tagging expert. Your task is to generate exactly {max_tags} relevant, descriptive tags for image prompts.

Rules:
1. Generate exactly {max_tags} tags
2. Tags should be single words or short phrases (2-3 words max)
3. Focus on: visual elements, mood, style, genre, setting, objects, characters, atmosphere
4. Use lowercase for consistency
5. Separate tags with commas
6. Be specific and relevant to the prompt content
7. Include both concrete elements (objects, places) and abstract concepts (mood, style)

Examples:
- For "Ancient dragon's lair with treasure": dragon, ancient, lair, treasure, fantasy, dark, mysterious, cave
- For "Romantic sunset garden": romantic, sunset, garden, flowers, peaceful, golden, nature, love

Return ONLY the tags, separated by commas, nothing else."""

            user_prompt = f"Generate tags for this image prompt: {prompt}"

            # Create prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt),
            ])

            # Generate tags using Azure OpenAI
            chain = prompt_template | self.llm | StrOutputParser()
            tags_text = chain.invoke({})
            
            tags = [tag.strip().lower() for tag in tags_text.split(',')]
            
            # Ensure we have exactly the requested number of tags
            if len(tags) > max_tags:
                tags = tags[:max_tags]
            elif len(tags) < max_tags:
                # If we have fewer tags, pad with generic ones
                generic_tags = ['art', 'illustration', 'digital', 'creative', 'detailed', 'atmospheric', 'cinematic', 'professional']
                for generic_tag in generic_tags:
                    if generic_tag not in tags and len(tags) < max_tags:
                        tags.append(generic_tag)
                tags = tags[:max_tags]
            
            return tags
            
        except Exception as e:
            print(f"Error generating tags for prompt '{prompt[:50]}...': {e}")
            # Return default tags if API fails
            return ['art', 'illustration', 'digital', 'creative', 'fantasy', 'detailed', 'atmospheric', 'cinematic']
    
    def process_json_file(self, filename: str, delay: float = 1.0) -> None:
        """Process a JSON file and add GPT-generated tags to each image reference"""
        print(f"Processing {filename}...")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"File {filename} not found, skipping...")
            return
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return
        
        total_items = 0
        processed_items = 0
        
        # Count total items first
        for category, items in data.items():
            if isinstance(items, list):
                total_items += len(items)
        
        # Process each category
        for category, items in data.items():
            if isinstance(items, list):
                print(f"Processing category: {category}")
                
                for i, item in enumerate(items):
                    if 'prompt' in item:
                        print(f"  Processing item {processed_items + 1}/{total_items}: {item['prompt'][:60]}...")
                        
                        # Generate tags using GPT
                        tags = self.generate_tags_for_prompt(item['prompt'])
                        item['tags'] = ', '.join(tags)
                        
                        processed_items += 1
                        
                        # Add delay to avoid rate limiting
                        if delay > 0:
                            time.sleep(delay)
                    else:
                        print(f"  Skipping item without prompt: {item}")
        
        # Save the updated data
        output_filename = filename.replace('.json', '_with_gpt_tags.json')
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Successfully processed {filename} -> {output_filename}")
            print(f"   Generated tags for {processed_items} items")
        except Exception as e:
            print(f"Error saving {output_filename}: {e}")

def main():
    """Main function to process all JSON files"""
    # Initialize the tag generator
    try:
        generator = GPTTagGenerator()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please set your Azure OpenAI credentials as environment variables:")
        print("  export AZURE_OPENAI_ENDPOINT='your-azure-endpoint'")
        print("  export AZURE_OPENAI_API_KEY='your-azure-api-key'")
        print("Or on Windows:")
        print("  set AZURE_OPENAI_ENDPOINT=your-azure-endpoint")
        print("  set AZURE_OPENAI_API_KEY=your-azure-api-key")
        return
    
    # Files to process
    files = [
        'chinese_kingdoms_data_20250805_103731.json',
        'drama_data_20250805_144158.json',
        'fantasy_image_data_20250805_091354.json',
        'romance_data_20250805_133414.json',
        'spooky_story_data_20250805_113814.json'
    ]
    
    print("🚀 Starting Azure OpenAI-powered tag generation...")
    print("⚠️  Note: This will make API calls to Azure OpenAI and may incur costs.")
    
    # Ask for confirmation
    response = input("Do you want to continue? (y/n): ").strip().lower()
    if response != 'y' and response != 'yes':
        print("❌ Operation cancelled.")
        return
    
    successful_files = 0
    total_files = len(files)
    
    for file in files:
        try:
            generator.process_json_file(file, delay=1.0)  # 1 second delay between API calls
            successful_files += 1
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
    
    print(f"\n🎉 Tag generation complete!")
    print(f"   Successfully processed {successful_files}/{total_files} files")
    
    if successful_files > 0:
        print(f"\n📁 Output files created:")
        for file in files:
            output_file = file.replace('.json', '_with_gpt_tags.json')
            if os.path.exists(output_file):
                print(f"   - {output_file}")

if __name__ == "__main__":
    main()