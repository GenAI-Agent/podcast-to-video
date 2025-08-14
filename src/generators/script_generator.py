#!/usr/bin/env python3
"""
Script Generator Module
Handles transcript generation from articles and web links
"""

import os
import sys
from typing import Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests

# Load environment variables
load_dotenv()

# GPT Prompts for script generation
TRANSCRIPT_PROMPT = """
You are a professional video speech writer. Based on the provided article, directly give me a speech draft of about 1 minute in length. The content should be concise, engaging, and suitable for spoken video narration.
Keep the original language of the article (Chinese or English); do not translate.
But when you use Chinese, you can only use Traditional Chinese.
"""

TRANSCRIPT_USER_PROMPT = """
Based on the following article, directly give me a 1-minute video speech draft:

{article}

Only provide the speech draft, do not include any other explanations.
Don't include any other text or symbols in your response.
"""

class ScriptGenerator:
    def __init__(self):
        """Initialize the script generator with Azure OpenAI"""
        self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-testing")
        
        if not self.api_base or not self.api_key:
            raise ValueError("Azure OpenAI credentials not found in environment variables")
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.api_base,
            api_key=self.api_key,
            api_version="2025-01-01-preview",
            azure_deployment=self.deployment_name,
            temperature=0.7,
            max_tokens=1500
        )
    
    def generate_from_text(self, article_text: str) -> str:
        """
        Generate a video script from article text
        
        Args:
            article_text: Raw article text
            
        Returns:
            Generated transcript suitable for video narration
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", TRANSCRIPT_PROMPT),
                ("user", TRANSCRIPT_USER_PROMPT)
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            transcript = chain.invoke({"article": article_text})
            
            return transcript.strip()
            
        except Exception as e:
            raise Exception(f"Failed to generate transcript: {str(e)}")
    
    def generate_from_url(self, web_url: str) -> str:
        """
        Generate a video script from a web URL
        
        Args:
            web_url: URL to extract content from
            
        Returns:
            Generated transcript suitable for video narration
        """
        try:
            # Extract content from URL
            article_text = self._extract_content_from_url(web_url)
            
            # Generate transcript from extracted content
            return self.generate_from_text(article_text)
            
        except Exception as e:
            raise Exception(f"Failed to generate transcript from URL: {str(e)}")
    
    def _extract_content_from_url(self, url: str) -> str:
        """Extract readable content from a web URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text from common content containers
            content_selectors = [
                'article', 'main', '.content', '.post-content', 
                '.entry-content', '.article-content', 'section'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text()
                    break
            
            # Fallback to body if no specific content found
            if not content:
                content = soup.get_text()
            
            # Clean up the text
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length (approximately 3000 characters for reasonable processing)
            if len(content) > 3000:
                content = content[:3000] + "..."
            
            return content
            
        except Exception as e:
            raise Exception(f"Failed to extract content from URL: {str(e)}")
    
    def validate_script(self, script: str) -> dict:
        """
        Validate a script for video generation
        
        Args:
            script: Script text to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            "valid": True,
            "warnings": [],
            "stats": {}
        }
        
        # Basic validation checks
        if not script or not script.strip():
            result["valid"] = False
            result["warnings"].append("Script is empty")
            return result
        
        # Calculate script statistics
        word_count = len(script.split())
        char_count = len(script)
        estimated_duration = word_count / 150 * 60  # Assume 150 words per minute
        
        result["stats"] = {
            "word_count": word_count,
            "character_count": char_count,
            "estimated_duration_seconds": round(estimated_duration, 2)
        }
        
        # Add warnings for script length
        if word_count < 50:
            result["warnings"].append("Script is quite short, may result in very brief video")
        elif word_count > 300:
            result["warnings"].append("Script is quite long, may result in lengthy video")
        
        if estimated_duration < 30:
            result["warnings"].append("Estimated duration is less than 30 seconds")
        elif estimated_duration > 120:
            result["warnings"].append("Estimated duration exceeds 2 minutes")
        
        return result