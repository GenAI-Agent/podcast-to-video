# Batch Image Data Processing System User Guide

A comprehensive system for processing batch image data, including data collection, tag extraction, database storage, and vector search functionality.

## 📋 Table of Contents
- [System Overview](#system-overview)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [Standard Workflow](#standard-workflow)
- [Advanced Features](#advanced-features)
- [Data Structure Reference](#data-structure-reference)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

## 🔍 System Overview

This system provides a complete image data processing and vector search solution:
1. **Data Import**: Import JSON-formatted image metadata into PostgreSQL
2. **Vector Indexing**: Create dual vector indexes based on prompts and descriptions
3. **Semantic Search**: Support multi-condition filtered vector similarity search
4. **Data Management**: Automatic synchronization, deduplication, and data cleaning

### Workflow
```
JSON Files → PostgreSQL → Pinecone (Dual Namespace) → Vector Search
```

## 🏗️ System Architecture

```
batch_image/
├── data/                           # Data directory
│   ├── json/                      # JSON format data files
│   ├── media/                     # Media files (audio, video)
│   └── text/                      # Text files
├── src/                           # Source code directory
│   ├── database/                  # Database-related modules
│   │   ├── pinecone_handler.py   # Pinecone vector database operations
│   │   └── postgres_handler.py   # PostgreSQL database operations
│   ├── scripts/                   # Core execution scripts
│   │   ├── import_json_to_postgres.py    # JSON data import
│   │   ├── upload_postgres_to_pinecone.py # Upload to vector database
│   │   └── delete_pinecone_data.py        # Delete vector data
│   ├── utils/                     # Utility functions
│   │   ├── add_file_paths.py     # Add file paths
│   │   └── retrieve_tags.py      # Tag extraction
│   ├── examples/                  # Usage examples
│   └── generators/                # Content generators
├── batch_image_data_*.json        # Batch image data files
└── workflows/                     # Workflow definitions
```

## 📦 Prerequisites

- Python 3.8+
- PostgreSQL database (with working connection)
- Pinecone account and API key
- Azure OpenAI API access
- Internet connection (for external API access)

## 🚀 Installation

1. Navigate to project directory:
```bash
cd batch_image
```

2. Create virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚙️ Environment Configuration

### Environment Variables Setup (create `.env` file)
```env
# Pinecone Vector Database
PINECONE_API_KEY=your_pinecone_api_key

# Azure OpenAI (for vector generation)
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint

# NVIDIA API (optional)
NVIDIA_API_KEY=your_nvidia_api_key
```

## 📖 Standard Workflow

### 1. Import Data to PostgreSQL
Import `batch_image_data_*_final.json` files from root directory to database:

```bash
python src/scripts/import_json_to_postgres.py
```

**Features:**
- Automatically searches for JSON files matching the pattern in root directory
- Parses JSON data and formats it into database records
- Handles duplicate records using `ON CONFLICT`
- Displays detailed import statistics

### 2. Upload to Pinecone Vector Database
Automatically upload to two namespaces for vector indexing:

```bash
python src/scripts/upload_postgres_to_pinecone.py
```

**Dual Namespace Architecture:**
- `prompt` namespace: Vector search based on prompt content
- `description` namespace: Vector search based on tags/description

**Key Features:**
- Automatically handles various prompt formats (including nested JSON)
- Batch upload for improved efficiency
- Automatic error handling and retry mechanisms
- Detailed upload progress and statistics

### 3. Data Query and Search

#### Python API Usage Examples:

```python
from src.database.pinecone_handler import PineconeHandler

handler = PineconeHandler()

# Semantic search based on prompts
results = handler.query_pinecone(
    query="bull market investment strategy",
    metadata_filter={"sub_theme": "BULLISH"},
    index_name="image-library",
    namespace="prompt",
    top_k=5
)

# Search based on description/tags
results = handler.query_pinecone(
    query="office building cityscape",
    metadata_filter={"theme": "lens_quant"},
    index_name="image-library", 
    namespace="description",
    top_k=5
)

# Hybrid search - multi-condition filtering
results = handler.query_pinecone(
    query="financial growth",
    metadata_filter={
        "$and": [
            {"sub_theme": {"$in": ["BULLISH", "WALLSTREET"]}},
            {"theme": {"$eq": "lens_quant"}}
        ]
    },
    index_name="image-library",
    namespace="prompt",
    top_k=10
)
```

## 🔧 Advanced Features

### Data Cleanup
Batch delete vectors from Pinecone based on conditions:

```bash
# Edit deletion conditions in src/scripts/delete_pinecone_data.py
python src/scripts/delete_pinecone_data.py
```

### File Path Management
Add file path information to JSON data:

```bash
python src/utils/add_file_paths.py
```

### Batch Tag Extraction
Use AI to batch extract image tags:

```bash
python src/utils/retrieve_tags.py
```

## 📊 Data Structure Reference

### PostgreSQL Table Structure (image_library)
```sql
CREATE TABLE image_library (
    id UUID PRIMARY KEY,                -- Unique identifier
    name VARCHAR(255),                  -- File name
    file_path TEXT,                     -- Complete file path
    prompt TEXT,                        -- Original prompt (JSON format)
    description TEXT,                   -- Tags and description
    theme VARCHAR(100),                 -- Theme classification (e.g., lens_quant)
    sub_theme VARCHAR(100),             -- Sub-theme (e.g., BULLISH, CRYPTO)
    status VARCHAR(50) DEFAULT 'active' -- Record status
);
```

### Pinecone Vector Record Structure
```python
{
    "id": "uuid-string",
    "values": [512-dimensional vector array],
    "metadata": {
        "name": "Image file name",
        "file_path": "Complete file path", 
        "description": "Tags and description content",
        "theme": "Theme classification",
        "sub_theme": "Sub-theme classification",
        "original_prompt": "First 500 characters of original prompt"
    }
}
```

### JSON Input Format Example
```json
{
  "BULLISH": [
    {
      "task_id": "uuid-here",
      "prompt": "{\"prompt\": {\"description\": \"bull market scene\"}}",
      "file_name": "batch_bullish_2025-08-04_14-25-02-0001",
      "tags": "finance, market, growth, success",
      "file_path": "C:\\path\\to\\image.png"
    }
  ]
}
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Database Connection Failed
```bash
# Error message: connection failed
```
**Resolution Steps:**
- Check network connectivity
- Verify database server is running
- Confirm connection parameters are correct (host, port, credentials)
- Check firewall and security group settings

#### 2. Pinecone Upload Failed
```bash
# Error message: Vector dimension 0 does not match
```
**Resolution Steps:**
- Verify AZURE_OPENAI_API_KEY validity
- Check network connection to Azure OpenAI
- Confirm vector dimension is set to 512
- Check if prompt extraction is working properly

#### 3. JSON Format Error
```bash
# Error message: JSONDecodeError
```
**Resolution Steps:**
- Use JSON validation tools to check file format
- Ensure file encoding is UTF-8
- Check for special characters or escaping issues

#### 4. Out of Memory Error
```bash
# Error message: MemoryError or OOM
```
**Resolution Steps:**
- Reduce batch_size parameter (default 50 → 25)
- Process large datasets in smaller batches
- Increase system virtual memory

### Debugging Techniques

#### 1. Check Data Synchronization Status
```python
from src.database.pinecone_handler import PineconeHandler

handler = PineconeHandler()

# Check PostgreSQL record count
data = handler.fetch_image_library_data()
print(f"PostgreSQL total records: {len(data)}")

# Check Pinecone vector statistics
index = handler._pc.Index("image-library")
stats = index.describe_index_stats()
for ns_name, ns_stats in stats.namespaces.items():
    print(f"{ns_name} namespace: {ns_stats.vector_count} vectors")
```

#### 2. Test Prompt Extraction Function
```python
# Test if prompt extraction is working
sample_prompt = '{"prompt": {"description": "test content"}}'
extracted = handler.extract_prompt_from_json(sample_prompt)
print(f"Extraction result type: {type(extracted)}")
print(f"Extraction content: {extracted}")
```

#### 3. Verify Vector Search
```python
# Test search functionality
test_results = handler.query_pinecone(
    query="test search",
    metadata_filter={},
    index_name="image-library",
    namespace="prompt",
    top_k=1
)
print(f"Search result count: {len(test_results)}")
```

## ⚡ Performance Optimization

### 1. Batch Processing Parameter Adjustment
```python
# Adjust batch size based on system resources
- Sufficient memory: batch_size=50-100
- Limited memory: batch_size=25-50
- Unstable network: batch_size=10-25
```

### 2. Network Optimization Strategy
- Choose geographically close database regions
- Use stable network connections
- Consider CDN or proxy servers

### 3. Data Preprocessing
- Ensure JSON format is correct
- Pre-validate required field completeness
- Remove unnecessary large files

### 4. Monitoring and Maintenance
```python
# Regular system status checks
import psutil
print(f"CPU usage: {psutil.cpu_percent()}%")
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

## 🔄 System Maintenance

### Regular Maintenance Tasks

#### 1. Data Backup
```bash
# PostgreSQL backup
pg_dump -h host -U user -d database > backup.sql

# Original file backup
tar -czf json_backup.tar.gz batch_image_data_*.json
```

#### 2. Performance Monitoring
- Monitor API usage and quotas
- Track vector search response times
- Check database query performance

#### 3. Data Cleanup
- Regularly clean invalid records
- Remove duplicate vector indexes
- Optimize database indexes

### Updates and Extensions

#### When Adding New Data
```bash
# Standard process
python src/scripts/import_json_to_postgres.py
python src/scripts/upload_postgres_to_pinecone.py
```

#### When Data Format Changes
- Update `extract_prompt_from_json` method
- Adjust database schema (if needed)
- Re-index existing data

## 📞 Technical Support

If you encounter technical issues, please troubleshoot in the following order:
1. Check system logs and error messages
2. Verify environment variables and credentials
3. Test network connectivity and API access
4. Refer to the troubleshooting section in this document
5. Check code comments in each module

For more technical details, please refer to the docstrings and code comments in the modules under the `src/` directory.