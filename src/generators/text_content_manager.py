#!/usr/bin/env python3
"""
Text Content Manager
Handles text processing, web content extraction, and transcript generation
"""

import os
import re
import requests
from typing import List
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Prompts for transcript generation
TRANSCRIPT_PROMPT = """
    You are a professional video speech writer. Based on the provided article, directly give me a speech draft of about 1 minute in length. The content should be concise, engaging, and suitable for spoken video narration.
    Keep the original language of the article (Chinese or English); do not translate.
    But when you use Chinese, you can only use Traditional Chinese.
"""

TRANSCRIPT_USER_PROMPT = """
    Based on the following article, directly give me a 1-minute video speech draft:

    {article}

    Only provide the speech draft, do not include any other explanations.
    Dont include any other text or symbols in your response.
    
"""


class TextContentManager:
    """Manager class for handling text content processing"""
    
    def __init__(self):
        """Initialize the text content manager"""
        pass
    
    def generate_transcript_from_article(self, article: str) -> str:
        """Generate a 1-minute transcript from an article using GPT"""
        try:
            # Get Azure OpenAI credentials from environment
            api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")

            if not api_base or not api_key:
                print(
                    "Warning: Azure OpenAI credentials not found. Returning original article."
                )
                return article

            # Initialize Azure OpenAI
            llm = AzureChatOpenAI(
                azure_endpoint=api_base,
                api_key=api_key,
                azure_deployment="gpt-4o-testing",
                api_version="2025-01-01-preview",
                temperature=0.7,
                max_tokens=500,
                timeout=None,
                max_retries=2,
            )

            # Create prompt template for transcript generation
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        TRANSCRIPT_PROMPT,
                    ),
                    (
                        "human",
                        TRANSCRIPT_USER_PROMPT,
                    ),
                ]
            )

            # Generate transcript
            chain = prompt_template | llm | StrOutputParser()
            transcript = chain.invoke({"article": article})

            print("✓ Successfully generated transcript using GPT")
            return transcript

        except Exception as e:
            print(f"Error generating transcript: {e}")
            print("Falling back to original article")
            return article

    def extract_web_content(self, web_link: str) -> str:
        """Extract text content from a web page

        Args:
            web_link: URL of the web page to extract content from

        Returns:
            str: Extracted text content from the web page
        """
        try:
            # Validate URL
            parsed_url = urlparse(web_link)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL: {web_link}")

            # Set headers to mimic a browser request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            # Fetch the web page
            print(f"Fetching content from: {web_link}")
            response = requests.get(web_link, headers=headers, timeout=30)
            response.raise_for_status()

            # Parse HTML content
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text from common content elements
            content_elements = soup.find_all(
                ["p", "h1", "h2", "h3", "h4", "h5", "h6", "article", "section", "main"]
            )

            # If no specific content elements found, get all text
            if not content_elements:
                text = soup.get_text()
            else:
                text_parts = []
                for element in content_elements:
                    element_text = element.get_text().strip()
                    if element_text:
                        text_parts.append(element_text)
                text = "\n\n".join(text_parts)

            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Ensure we have meaningful content
            if len(text) < 100:
                raise ValueError(
                    "Extracted content is too short. The page might be dynamic or protected."
                )

            print(f"✓ Successfully extracted {len(text)} characters from web page")
            return text

        except requests.RequestException as e:
            print(f"Error fetching web page: {e}")
            raise
        except Exception as e:
            print(f"Error extracting web content: {e}")
            raise

    def split_article_into_sentences(self, article: str, max_chars: int = 25) -> List[str]:
        """Split article into short sentences without punctuation
        
        Args:
            article: The article text to split
            max_chars: Maximum characters per sentence (default: 25)
            
        Returns:
            List of sentences
        """
        # Clean up the article and replace commas with spaces
        article = article.strip()
        article = re.sub(r"[,，]", " ", article)  # Replace commas with spaces
        article = re.sub(r"\s+", " ", article)  # Clean up multiple spaces

        # Split by sentence endings, including Chinese punctuation
        sentences = re.split(r"[.!?。！？]+", article)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Further split sentences to ensure max characters
        final_sentences = []

        for sentence in sentences:
            # Keep the sentence as is initially, don't strip punctuation yet
            sentence = sentence.strip()

            if len(sentence) <= max_chars:
                if sentence:  # Only add non-empty sentences
                    final_sentences.append(sentence)
            else:
                # Split long sentences into chunks of max characters
                # Try to split at natural word boundaries while preserving paired punctuation
                words = sentence.split()
                current_chunk = ""

                for word in words:
                    # Check if adding this word would exceed limit
                    test_chunk = current_chunk + (" " if current_chunk else "") + word

                    if len(test_chunk) <= max_chars:
                        current_chunk = test_chunk
                    else:
                        # Save current chunk if not empty and meaningful
                        if current_chunk and len(current_chunk.strip()) > 0:
                            final_sentences.append(current_chunk)

                        # Start new chunk with current word
                        if len(word) > max_chars:
                            # If single word is too long, check if it contains paired punctuation
                            # Keep book titles and parentheses together
                            if any(
                                p in word for p in ["《》", "「」", "（）", "()", '""']
                            ):
                                final_sentences.append(
                                    word
                                )  # Keep paired punctuation together
                            else:
                                for i in range(0, len(word), max_chars):
                                    chunk = word[i : i + max_chars]
                                    if chunk and chunk.strip():
                                        final_sentences.append(chunk)
                            current_chunk = ""
                        else:
                            current_chunk = word

                # Add remaining chunk
                if current_chunk and current_chunk.strip():
                    final_sentences.append(current_chunk)

        # Final cleanup: merge standalone punctuation with adjacent sentences
        cleaned_sentences = []
        for i, sentence in enumerate(final_sentences):
            sentence = sentence.strip()
            # Check if this is just standalone punctuation
            content_without_punct = sentence.strip(",.，。、；;：:》」）)")

            if len(content_without_punct) == 0:  # Just punctuation
                # Try to merge with previous sentence if possible
                if (
                    cleaned_sentences
                    and len(cleaned_sentences[-1]) + len(sentence) <= max_chars
                ):
                    cleaned_sentences[-1] += sentence
                # Otherwise skip standalone punctuation
            elif len(sentence) <= 2 and i > 0:  # Very short sentence, try to merge
                if len(cleaned_sentences[-1]) + len(sentence) + 1 <= max_chars:
                    cleaned_sentences[-1] += sentence
                else:
                    cleaned_sentences.append(sentence)
            else:
                if sentence:  # Only add non-empty sentences
                    cleaned_sentences.append(sentence)

        return cleaned_sentences