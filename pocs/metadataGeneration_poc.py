import re
import json
import csv
from datasets import load_dataset
from collections import defaultdict

def extract_table_instances_from_hf_dataset(dataset_name: str, split: str = "train") -> list:
    """
    Loads a Hugging Face dataset, extracts SQL schemas from 'sql_context' field,
    and returns details for each table instance found, including its domain, sql_context,
    column names, and data types.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face (e.g., "gretelai/synthetic_text_to_sql").
        split (str): The dataset split to analyze (e.g., "train", "test", "validation").

    Returns:
        list: A list of dictionaries, each containing 'table_name', 'domain', 'sql_context',
              'column_name', and 'data_type' for each column found within a CREATE TABLE statement.
              Duplicate records are included if a table appears multiple times or has multiple columns.
    """
    print(f"Attempting to load dataset: '{dataset_name}', split: '{split}'...")
    try:
        dataset = load_dataset(dataset_name, split=split)
        print(f"Dataset loaded successfully. Number of records in '{split}' split: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset name and split are correct, and you have an internet connection.")
        return [] # Return an empty list if dataset loading fails

    extracted_instances = [] # List to store all extracted instances

    # Regex to find CREATE TABLE statements and extract table name and column definitions
    create_table_regex = re.compile(
        r"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?`?(\w+)`?\s*\((.*?)\);",
        re.DOTALL | re.IGNORECASE
    )

    # Regex to find column names and their data types within the column definitions.
    # It handles optional backticks, various data type formats (e.g., VARCHAR(50), DECIMAL(10,2)),
    # and common column constraints (PRIMARY KEY, UNIQUE, NOT NULL, DEFAULT).
    column_regex = re.compile(
        r"`?(\w+)`?\s+([A-Z]+(?:(?:\s*\(\d+(?:,\s*\d+)?\))|\s*\(.*?\))*\s*(?:(?:PRIMARY\s+KEY|UNIQUE|NOT\s+NULL|DEFAULT\s+.*?)(?:\s+|$))*?)",
        re.IGNORECASE
    )

    for i, record in enumerate(dataset):
        sql_context = record.get('sql_context', '')
        domain = record.get('domain', '') # Capture domain for the current record
        if not sql_context:
            continue # Skip records without sql_context

        # Find all CREATE TABLE statements in the current record's sql_context
        for table_match in create_table_regex.finditer(sql_context):
            table_name = table_match.group(1)
            columns_definition_str = table_match.group(2)
            
            # Robustly split columns_definition_str by comma, respecting parentheses balance.
            column_definitions = []
            balance = 0
            current_col_chars = []
            for char in columns_definition_str:
                if char == '(':
                    balance += 1
                elif char == ')':
                    balance -= 1
                elif char == ',' and balance == 0:
                    column_definitions.append("".join(current_col_chars).strip())
                    current_col_chars = []
                    continue
                current_col_chars.append(char)
            if current_col_chars: # Add the last column definition
                column_definitions.append("".join(current_col_chars).strip())

            # If no columns are found for a table, add a record with empty column/data_type
            if not column_definitions:
                extracted_instances.append({
                    "domain": domain,
                    "table_name": table_name,
                    "column_name": "",
                    "data_type": "",
                    "sql_context": sql_context
                })
            else:
                for col_def in column_definitions:
                    if not col_def:
                        continue
                    col_match = column_regex.match(col_def)
                    if col_match:
                        col_name = col_match.group(1)
                        data_type = re.sub(r'\s+', ' ', col_match.group(2)).strip()
                        
                        # Append the details for each column
                        extracted_instances.append({
                            "domain": domain,
                            "table_name": table_name,
                            "column_name": col_name,
                            "data_type": data_type,
                            "sql_context": sql_context
                        })
    
    return extracted_instances

if __name__ == "__main__":
    dataset_to_analyze = "gretelai/synthetic_text_to_sql"
    split_to_analyze = "train"

    print(f"--- Starting Analysis of '{dataset_to_analyze}' ({split_to_analyze} split) ---")
    
    # Call the function to extract all instances
    extracted_data = extract_table_instances_from_hf_dataset(dataset_to_analyze, split=split_to_analyze)

    if extracted_data:
        output_csv_filename = f"table_metadata.csv"
        
        # Define CSV headers to include new column and data type fields
        fieldnames = ['domain', 'table_name', 'column_name', 'data_type', 'sql_context']

        try:
            with open(output_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader() # Write the header row
                writer.writerows(extracted_data) # Write all data rows
            print(f"\nSuccessfully stored {len(extracted_data)} records to '{output_csv_filename}' in CSV format.")
        except Exception as e:
            print(f"\nError saving data to CSV: {e}")
    else:
        print("No table instances could be extracted or dataset is empty/unavailable.")

    print("\n--- Analysis Complete ---")