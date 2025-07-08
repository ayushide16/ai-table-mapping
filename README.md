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

### 🔹 `app.py`
The **main entry point** for running the complete workflow. Executes the full feature from prompt to mapping.

### 🔹 `backend.py`
Handles the **core logic**, including:
- Interacting with Google Gemini API
- Processing LLM responses
- Parsing responses in `.json` format

### 🔹 `llm_prompt.py`
Defines the **prompts sent to the LLM** (Gemini) for SQL query generation and metadata mapping.

### 🔹 `pocs/`
Contains **proof-of-concept (POC)** experiments:
- `prompt_poc.py`: Evaluates the effectiveness of different prompt strategies.
- `metadataGeneration_poc.py`: Generates metadata for the base dataset using the [Gretel AI Synthetic SQL Dataset](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql).

---

## Current Status

✅ Successfully generates SQL queries from partner data using LLM (Gemini).  
⚠️ **Ongoing challenge**: Robust and scalable **metadata generation** from partner datasets.

---

## Dataset Used

- [Gretel AI Synthetic Text-to-SQL Dataset](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)  
Used for metadata creation and benchmarking prompt performance.
