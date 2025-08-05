# Image Search and Format Tool

This tool allows you to search through your batch image data JSON file for keywords and generate formatted output suitable for video editing or other purposes.

## Files

- `search_and_format.py` - Main script with interactive menu and ImageSearcher class
- `search_example.py` - Simple example showing programmatic usage
- `batch_image_data_20250731_162750.json` - Your source data file

## Usage

### Interactive Mode

Run the main script for an interactive experience:

```bash
python search_and_format.py
```

This will present you with options to:
1. Search by keywords across all fields
2. Search in specific fields (prompt, tags, file_name, task_id)
3. Exit

### Programmatic Usage

You can also use the `ImageSearcher` class directly in your own scripts:

```python
from search_and_format import ImageSearcher

# Initialize searcher
searcher = ImageSearcher("batch_image_data_20250731_162750.json")

# Search for keywords and generate output
result = searcher.search_and_format(
    keywords=["trading", "monitor"],
    duration=3,
    output_file="output.txt"
)
```

### Run Example

See `search_example.py` for various usage examples:

```bash
python search_example.py
```

## Output Format

The tool generates output in the format you requested:

```
file 'image1.png'
duration 3
file 'image2.png'
duration 3
file 'image3.png'
duration 3
```

## Search Options

- **Keywords**: Comma-separated list of terms to search for
- **Duration**: How long each image should be displayed (default: 3)
- **Search Fields**: Choose which fields to search in:
  - `prompt` - The detailed image description
  - `tags` - Associated tags
  - `file_name` - The base filename
  - `task_id` - Unique identifier
- **Output File**: Specify a filename to save results, or leave blank for console output

## Examples

### Search for trading-related images:
```
Keywords: trading, desk, monitor
Duration: 3
Output: trading_images.txt
```

### Search only in tags for coffee images:
```
Keywords: coffee, mug
Fields: tags
Duration: 5
Output: coffee_images.txt
```

### Search for cinematic content:
```
Keywords: cinematic, professional
Fields: prompt
Duration: 4
Output: cinematic_images.txt
```

The tool will show you how many matches were found and which keywords matched for each result. 
