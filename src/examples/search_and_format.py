#!/usr/bin/env python3
"""
Search batch image data JSON file for keywords and generate formatted output.
"""

import json
import re
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


class ImageSearcher:
    def __init__(self, json_file_path: str):
        """Initialize the image searcher with a JSON file."""
        self.json_file_path = json_file_path
        self.data = self._load_json_data()
    
    def _load_json_data(self) -> Dict[str, Any]:
        """Load and parse the JSON data file."""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: File '{self.json_file_path}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in file '{self.json_file_path}': {e}")
            sys.exit(1)
    
    def search_keywords(self, keywords: List[str], search_in: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for keywords in the JSON data.
        
        Args:
            keywords: List of keywords to search for
            search_in: List of fields to search in ['prompt', 'tags', 'file_name', 'task_id']
                      If None, searches in all fields
        
        Returns:
            List of matching entries with metadata
        """
        if search_in is None:
            search_in = ['prompt', 'tags', 'file_name', 'task_id']
        
        matches = []
        image_counter = 1
        
        # Convert keywords to lowercase for case-insensitive search
        keywords_lower = [keyword.lower() for keyword in keywords]
        
        # Search through all categories
        for category_name, category_data in self.data.items():
            if isinstance(category_data, list):
                for entry in category_data:
                    if self._entry_matches_keywords(entry, keywords_lower, search_in):
                        match_info = {
                            'category': category_name,
                            'task_id': entry.get('task_id', ''),
                            'file_name': entry.get('file_name', ''),
                            'image_number': image_counter,
                            'matched_keywords': self._get_matched_keywords(entry, keywords_lower, search_in)
                        }
                        matches.append(match_info)
                        image_counter += 1
        
        return matches
    
    def _entry_matches_keywords(self, entry: Dict[str, Any], keywords_lower: List[str], search_in: List[str]) -> bool:
        """Check if an entry matches any of the keywords."""
        search_text = ""
        
        # Build search text from specified fields
        for field in search_in:
            if field in entry and entry[field]:
                search_text += str(entry[field]).lower() + " "
        
        # Check if any keyword is found in the search text
        return any(keyword in search_text for keyword in keywords_lower)
    
    def _get_matched_keywords(self, entry: Dict[str, Any], keywords_lower: List[str], search_in: List[str]) -> List[str]:
        """Get list of keywords that matched in this entry."""
        search_text = ""
        for field in search_in:
            if field in entry and entry[field]:
                search_text += str(entry[field]).lower() + " "
        
        return [keyword for keyword in keywords_lower if keyword in search_text]
    
    def generate_output_format(self, matches: List[Dict[str, Any]], duration: int = 3, 
                             output_file: str = None, image_extension: str = 'png') -> str:
        """
        Generate output in the specified format.
        
        Args:
            matches: List of matching entries
            duration: Duration for each image (default: 3)
            output_file: File to write output to (if None, returns string)
            image_extension: File extension for images (default: 'png')
        
        Returns:
            Formatted output string
        """
        output_lines = []
        
        for match in matches:
            image_filename = f"image{match['image_number']}.{image_extension}"
            output_lines.append(f"file '{image_filename}'")
            output_lines.append(f"duration {duration}")
        
        output_text = '\n'.join(output_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(output_text)
            print(f"Output written to '{output_file}'")
        
        return output_text
    
    def search_and_format(self, keywords: List[str], duration: int = 3, 
                         output_file: str = None, search_in: List[str] = None,
                         image_extension: str = 'png') -> str:
        """
        Search for keywords and return formatted output.
        
        Args:
            keywords: List of keywords to search for
            duration: Duration for each image
            output_file: File to write output to
            search_in: Fields to search in
            image_extension: File extension for images
        
        Returns:
            Formatted output string
        """
        matches = self.search_keywords(keywords, search_in)
        
        if not matches:
            message = f"No matches found for keywords: {', '.join(keywords)}"
            print(message)
            return message
        
        print(f"Found {len(matches)} matches for keywords: {', '.join(keywords)}")
        
        # Print summary of matches
        for match in matches:
            print(f"  - Image {match['image_number']}: {match['category']} "
                  f"(matched: {', '.join(match['matched_keywords'])})")
        
        return self.generate_output_format(matches, duration, output_file, image_extension)


def main():
    """Main function to demonstrate usage."""
    # Default JSON file
    json_file = "data/json/batch_image_data_20250731_162750.json"
    
    # Check if file exists
    if not Path(json_file).exists():
        print(f"Error: JSON file '{json_file}' not found.")
        print("Please ensure the file is in the current directory.")
        return
    
    # Initialize searcher
    searcher = ImageSearcher(json_file)
    
    # Example usage
    print("=== Image Search and Format Tool ===\n")
    
    # Interactive mode
    while True:
        print("\nOptions:")
        print("1. Search by keywords")
        print("2. Search in specific fields")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '3':
            print("Goodbye!")
            break
        elif choice == '1':
            keywords_input = input("Enter keywords (comma-separated): ").strip()
            if not keywords_input:
                print("No keywords provided.")
                continue
            
            keywords = [kw.strip() for kw in keywords_input.split(',')]
            duration = input("Enter duration (default: 3): ").strip()
            duration = int(duration) if duration.isdigit() else 3
            
            output_file = input("Enter output filename (press Enter for console output): ").strip()
            output_file = output_file if output_file else None
            
            result = searcher.search_and_format(keywords, duration, output_file)
            
            if not output_file:
                print("\n=== Generated Output ===")
                print(result)
        
        elif choice == '2':
            keywords_input = input("Enter keywords (comma-separated): ").strip()
            if not keywords_input:
                print("No keywords provided.")
                continue
            
            keywords = [kw.strip() for kw in keywords_input.split(',')]
            
            print("Available fields: prompt, tags, file_name, task_id")
            fields_input = input("Enter fields to search in (comma-separated, or press Enter for all): ").strip()
            search_in = [f.strip() for f in fields_input.split(',')] if fields_input else None
            
            duration = input("Enter duration (default: 3): ").strip()
            duration = int(duration) if duration.isdigit() else 3
            
            output_file = input("Enter output filename (press Enter for console output): ").strip()
            output_file = output_file if output_file else None
            
            result = searcher.search_and_format(keywords, duration, output_file, search_in)
            
            if not output_file:
                print("\n=== Generated Output ===")
                print(result)
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main() 