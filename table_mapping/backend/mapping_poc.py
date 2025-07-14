import os
import sys

# Ensure the parent directory (table_mapping) is in the Python path
# so that rag_model and config can be imported correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now, import modules from the backend package
from rag_model import RAGSchemaMapper
from config import GEMINI_API_KEY # Assuming you're fetching the key from here

# Define the path to your table_metadata.csv
# Assuming it's one level up from the 'backend' directory
METADATA_CSV_PATH = "../table_metadata.csv"

def run_poc():
    """Initializes the RAGSchemaMapper and runs a sample schema mapping using DDL."""
    print("--- Starting PoC for DDL-based RAG Schema Mapping ---")

    # 1. Initialize the Mapper
    try:
        # The persist_dir is relative to where the RAGSchemaMapper instance is created,
        # which will be within the backend directory from where this script runs.
        mapper = RAGSchemaMapper(
            metadata_path=METADATA_CSV_PATH,
            gemini_api_key=GEMINI_API_KEY,
            persist_dir="./data" # This will create/use backend/data/
        )
        print("\n--- RAGSchemaMapper Initialized Successfully ---")
    except Exception as e:
        print(f"Failed to initialize RAGSchemaMapper: {e}")
        print("Please ensure table_metadata.csv is in the correct path and GEMINI_API_KEY is set in config.py or .env.")
        return

    # 2. Define Sample Partner Schema Data (DDL)
    sample_ddl_input = {
        "partner_domain_input": "forestry",
        "ddl_statements": """
CREATE TABLE Forest_Workers (
    WorkerID INT PRIMARY KEY,
    FullName VARCHAR(255) NOT NULL,
    DateOfBirth DATE,
    Role ENUM('Logger', 'Forester', 'Supervisor'),
    ExperienceYears INT
);

CREATE TABLE Forest_Stands (
    StandID INT PRIMARY KEY,
    Location GEOMETRY,
    TreeSpecies VARCHAR(100),
    AgeYears INT,
    EstimatedVolumeCubicMeters DECIMAL(10, 2)
);

CREATE TABLE Sensor_Readings (
    ReadingID INT PRIMARY KEY,
    StandID INT,
    Timestamp DATETIME,
    Temperature DECIMAL(5,2),
    Humidity DECIMAL(5,2),
    SensorType VARCHAR(50)
);
"""
    }
    print(f"\n--- Attempting to map partner schema for domain: {sample_ddl_input['partner_domain_input']} ---")

    # 3. Call the New Schema Mapping Logic
    try:
        mapping_results = mapper.map_partner_schema(
            partner_domain_input=sample_ddl_input["partner_domain_input"],
            ddl_statements=sample_ddl_input["ddl_statements"]
        )

        # 4. Print the Results (Iterate through the list of mappings)
        print("\n--- Mapping Results ---")
        if not mapping_results:
            print("No mappings returned. Check the DDL or potential parsing issues.")
        else:
            for i, result in enumerate(mapping_results):
                print(f"\n--- Mapping {i+1} ---")
                if result.get("mapped_base_column"):
                    print(f"  Partner Column: `{result.get('partner_table')}.{result.get('partner_column')}` ({result.get('partner_data_type')})")
                    print(f"  Mapped Base Column: `{result.get('mapped_base_domain')}.{result.get('mapped_base_table')}.{result.get('mapped_base_column')}` ({result.get('mapped_base_data_type')})")
                    print(f"  FAISS Confidence Score: {result.get('confidence_score', 0.0):.2f}")
                    print(f"  LLM Explanation: {result.get('llm_explanation', 'No explanation provided.')}")
                else:
                    print(f"  No direct match found or an issue occurred for partner column: `{result.get('partner_table', 'N/A')}.{result.get('partner_column', 'N/A')}` ({result.get('partner_data_type', 'N/A')})")
                    print(f"  Reason: {result.get('llm_explanation', result.get('message', 'Unknown reason.'))}")
                
        print("\n--- Raw Backend Response (for debugging) ---")
        # You might want to uncomment this for very detailed debugging, but it can be long
        # import json
        # print(json.dumps(mapping_results, indent=2))

    except Exception as e:
        print(f"An error occurred during mapping: {e}")

if __name__ == "__main__":
    run_poc()