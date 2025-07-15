# AI Table Mapping

## Problem Statement

This project provides a **generalized AI-based table mapping workflow** for aligning and integrating large tabular datasets.

### Goal

Given:
- A **base dataset** with:
  - Table names  
  - Column names  
  - Column descriptions  
  - Table relationships  
  - Sample data

And:
- A **partner dataset** with:
  - One or more tables  
  - Column names (sometimes with descriptions)  
  - Relationship info (partial or full)  

The objective is to **map the base schema to the partner schema**, potentially merging or realigning tables, even when the structures vary. This is accomplished using LLM-assisted metadata analysis and SQL query generation.

---

## Repository Structure

### 🔹 `query_generator/app.py`  
The **main entry point** for running the complete workflow. Executes the full feature from prompt to mapping.

### 🔹 `query_generator/backend.py`  
Handles the **core logic**, including:
- Interacting with Google Gemini API
- Processing LLM responses
- Parsing responses in `.json` format

### 🔹 `query_generator/llm_prompt.py`  
Defines the **prompts sent to the LLM** (Gemini) for SQL query generation and metadata mapping.

### 🔹 `query_generator/pocs/`  
Contains **proof-of-concept (POC)** experiments:
- `prompt_poc.py`: Evaluates the effectiveness of different prompt strategies.
- `metadataGeneration_poc.py`: ✅ *Effectively generates metadata now.* Extracts:
  - Domain name  
  - Table name(s)  
  - Column name(s)  
  - Data type(s)  

### 🔹 `table_mapping/`  
🔧 Currently under development. This module will handle the actual **schema mapping** between the base and partner datasets.

---

## Mapping Workflow (RAG-Inspired)

The overall process follows a RAG-style (Retrieval-Augmented Generation) workflow:

### Step 1: Data Chunking  
Break down metadata into meaningful chunks.

### Step 3: Document Embeddings  
Transform metadata chunks into vector embeddings using embedding models.

### Step 4: Handling User Queries / Mapping Requests  
Convert schema-level or query-level requests into embeddings and find the most relevant schema components from the partner dataset.

### Step 5: Response Generation with LLM  
Use retrieved schema matches and metadata as context and generate:
- Mapped schema suggestions  
- SQL queries  
- Natural language explanations of mappings  

---

## Dataset Used

- [Gretel AI Synthetic Text-to-SQL Dataset](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)  
Used for metadata generation and prompt benchmarking.

---

## Current Status

✅ Information extraction from user input for partner schema  
✅ Accurate metadata extraction from metadata  
🚧 Schema mapping module in progress (`table_mapping/`)  
