#!/usr/bin/env python3
"""
Simple example of using the ImageSearcher to find matching images.
"""

import os
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.examples.search_and_format import ImageSearcher

def main():
    # Initialize the searcher
    searcher = ImageSearcher("data/json/batch_image_data_20250731_162750.json")
    
    # Example 1: Search for trading-related images
    print("=== Example 1: Search for trading-related content ===")
    keywords = ["trading", "monitor", "desk"]
    result = searcher.search_and_format(
        keywords=keywords,
        duration=3,
        output_file="trading_images.txt"
    )
    
    # Example 2: Search only in tags for coffee-related images
    print("\n=== Example 2: Search for coffee-related images in tags only ===")
    keywords = ["coffee", "mug"]
    result = searcher.search_and_format(
        keywords=keywords,
        duration=5,
        search_in=["tags"],
        output_file="coffee_images.txt"
    )
    
    # Example 3: Search for professional/cinematic content
    print("\n=== Example 3: Search for professional/cinematic content ===")
    keywords = ["cinematic", "professional"]
    result = searcher.search_and_format(
        keywords=keywords,
        duration=4,
        search_in=["prompt"],
        output_file="cinematic_images.txt"
    )
    
    # Example 4: Just print results without saving to file
    print("\n=== Example 4: Search for finance-related content (console output) ===")
    keywords = ["finance", "savings"]
    result = searcher.search_and_format(
        keywords=keywords,
        duration=3
    )
    print(result)

if __name__ == "__main__":
    main() 