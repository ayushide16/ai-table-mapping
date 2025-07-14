import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- 1. Load Your Base Metadata from CSV ---
try:
    base_metadata_df = pd.read_csv("table_metadata.csv")
    # Remove sql_context if it exists, as per your request.
    if 'sql_context' in base_metadata_df.columns:
        base_metadata_df = base_metadata_df.drop(columns=['sql_context'])
    print("--- Base Metadata Loaded from table_metadata.csv ---")
    print(base_metadata_df.head())
    print("\n")

except FileNotFoundError:
    print("Error: 'table_metadata.csv' not found. Please make sure the file is in the same directory as this script.")
    # Exit the script if the file is not found, as it's crucial for the workflow.
    exit()
except Exception as e:
    print(f"An error occurred while reading the CSV: {e}")
    exit()

# --- 2. Initialize the Embedding Model ---
# We use a pre-trained Sentence Transformer model.
# 'all-MiniLM-L6-v2' is a good balance of speed and performance.
model = SentenceTransformer('all-MiniLM-L6-v2')
print("--- Sentence Transformer Model Loaded ---")

# --- 3. Prepare Text for Embeddings ---
# We combine the relevant columns into a single string for each entry.
# This string is what the embedding model will "understand."
base_metadata_df['embedding_text'] = base_metadata_df.apply(
    lambda row: f"Domain: {row['domain']}. Table: {row['table_name']}. Column: {row['column_name']}. Data Type: {row['data_type']}.",
    axis=1
)

print("\n--- Example Embedding Text Samples ---")
print(base_metadata_df['embedding_text'].sample(2).tolist()) # Show a couple of examples
print("\n")

# --- 4. Generate Embeddings for all Base Metadata ---
print("--- Generating Embeddings for Base Metadata (This might take a moment)... ---")
base_embeddings = model.encode(base_metadata_df['embedding_text'].tolist(), show_progress_bar=True)
embedding_dimension = base_embeddings.shape[1] # Get the dimension of the embeddings

print(f"Generated {len(base_embeddings)} embeddings with dimension {embedding_dimension}")

# --- 5. Create a FAISS Index (Super-Fast Index) ---
# We'll use an IndexFlatIP for dot product similarity (equivalent to cosine similarity for normalized vectors)
index = faiss.IndexFlatIP(embedding_dimension)
# Add the embeddings to the index
index.add(base_embeddings)

print("\n--- FAISS Index Created and Populated ---")
print(f"Number of vectors in index: {index.ntotal}")

# --- Helper Function for Data Type Compatibility (from previous discussion) ---
def are_data_types_compatible(type1, type2):
    type1 = type1.lower().split('(')[0] # Handle VARCHAR(X) -> varchar
    type2 = type2.lower().split('(')[0]

    if type1 == type2:
        return True
    # Broad numerical compatibility
    num_types = ["int", "real", "numeric", "decimal", "float"]
    if any(n_type in type1 for n_type in num_types) and any(n_type in type2 for n_type in num_types):
        return True
    # Broad text compatibility
    text_types = ["char", "text", "varchar", "string"]
    if any(t_type in type1 for t_type in text_types) and any(t_type in type2 for t_type in text_types):
        return True
    # Date/Time compatibility
    date_types = ["date", "time", "timestamp"]
    if any(d_type in type1 for d_type in date_types) and any(d_type in type2 for d_type in date_types):
        return True
    return False

# --- Main Mapping Function ---
def map_partner_column(
    partner_domain_input: str,
    partner_table_name: str,
    partner_column_name: str,
    partner_data_type: str,
    base_metadata_df: pd.DataFrame,
    faiss_index: faiss.Index,
    embedding_model: SentenceTransformer,
    top_k: int = 5, # How many top candidates to retrieve
    min_confidence: float = 0.6 # Minimum score to consider a valid match
):
    """
    Maps a single partner column to the best matching base column using RAG.
    """
    print(f"\n--- Mapping Partner Column: {partner_domain_input} / {partner_table_name} / {partner_column_name} ({partner_data_type}) ---")

    # 1. Prepare Query Text
    partner_query_text = f"Domain: {partner_domain_input}. Table: {partner_table_name}. Column: {partner_column_name}. Data Type: {partner_data_type}."

    # 2. Generate Query Embedding
    query_embedding = embedding_model.encode([partner_query_text])

    # 3. Filter Base Metadata by User-Provided Domain
    # Get the indices of rows in base_metadata_df that match the domain
    domain_filtered_indices = base_metadata_df[base_metadata_df['domain'].str.lower() == partner_domain_input.lower()].index.values

    if len(domain_filtered_indices) == 0:
        print(f"No base metadata found for domain: '{partner_domain_input}'. Cannot map.")
        return None

    # Perform similarity search on the full index
    distances, all_indices = faiss_index.search(query_embedding, top_k * 5) # Retrieve more to ensure we find domain-matched ones

    # Filter results by domain and data type compatibilityo
    relevant_matches = []
    for i, idx in enumerate(all_indices[0]):
        if idx in domain_filtered_indices: # Check if the retrieved index is within the allowed domain
            base_row = base_metadata_df.iloc[idx]
            if are_data_types_compatible(partner_data_type, base_row['data_type']):
                relevant_matches.append({
                    "base_idx": idx,
                    "score": distances[0][i], # This is cosine similarity if vectors are normalized
                    "base_domain": base_row['domain'],
                    "base_table_name": base_row['table_name'],
                    "base_column_name": base_row['column_name'],
                    "base_data_type": base_row['data_type']
                })
        if len(relevant_matches) >= top_k: # Stop once we have enough domain-filtered, type-compatible matches
            break

    # Sort final relevant matches by score (highest first)
    relevant_matches.sort(key=lambda x: x['score'], reverse=True)

    if not relevant_matches:
        print("No suitable matches found after filtering by domain and data type compatibility.")
        return None

    best_match = relevant_matches[0]
    if best_match['score'] < min_confidence:
        print(f"Best match found (score: {best_match['score']:.2f}) but below minimum confidence threshold ({min_confidence}).")
        return None

    print(f"Found best match (score: {best_match['score']:.2f}):")
    print(f"  Base: {best_match['base_domain']} / {best_match['base_table_name']} / {best_match['base_column_name']} ({best_match['base_data_type']})")

    return best_match

# --- Define the partner data schema (simplified for demonstration) ---
# In a real scenario, this would come from an uploaded file or API.
partner_data_schema = [
    {
        "table_name": "forest_workers",
        "columns": [
            {"column_name": "person_id", "data_type": "INT"},
            {"column_name": "full_name", "data_type": "TEXT"},
            {"column_name": "territory", "data_type": "VARCHAR(100)"}
        ]
    },
    {
        "table_name": "lumber_sales",
        "columns": [
            {"column_name": "sale_record_id", "data_type": "INT"},
            {"column_name": "worker_id", "data_type": "INT"}, # This should map to salesperson_id
            {"column_name": "timber_quantity", "data_type": "REAL"},
            {"column_name": "date_of_sale", "data_type": "DATE"}
        ]
    },
    {
        "table_name": "machine_stats", # A table outside forestry, for testing domain filtering
        "columns": [
            {"column_name": "machine_type", "data_type": "VARCHAR(50)"},
            {"column_name": "last_maintenance_date", "data_type": "DATE"}
        ]
    }
]

# --- Simulate User Input for Partner Domain ---
user_provided_partner_domain = "forestry" # User tells us this is a forestry dataset

# --- Perform Mapping for Each Partner Column ---
mapped_results = []
for table in partner_data_schema:
    current_partner_table_name = table['table_name']
    for col in table['columns']:
        partner_col_name = col['column_name']
        partner_col_type = col['data_type']

        result = map_partner_column(
            partner_domain_input=user_provided_partner_domain,
            partner_table_name=current_partner_table_name,
            partner_column_name=partner_col_name,
            partner_data_type=partner_col_type,
            base_metadata_df=base_metadata_df, # Your pre-loaded base metadata
            faiss_index=index,                 # Your pre-built FAISS index
            embedding_model=model              # Your pre-loaded Sentence Transformer model
        )
        if result:
            mapped_results.append({
                "partner_domain": user_provided_partner_domain,
                "partner_table": current_partner_table_name,
                "partner_column": partner_col_name,
                "partner_data_type": partner_col_type,
                "mapped_base_domain": result['base_domain'],
                "mapped_base_table": result['base_table_name'],
                "mapped_base_column": result['base_column_name'],
                "mapped_base_data_type": result['base_data_type'],
                "confidence_score": result['score']
            })
        else:
            mapped_results.append({
                "partner_domain": user_provided_partner_domain,
                "partner_table": current_partner_table_name,
                "partner_column": partner_col_name,
                "partner_data_type": partner_col_type,
                "mapped_base_domain": None,
                "mapped_base_table": None,
                "mapped_base_column": None,
                "mapped_base_data_type": None,
                "confidence_score": 0.0 # Indicate no match
            })

print("\n\n--- Final Mapping Summary ---")
for res in mapped_results:
    print(f"Partner: {res['partner_domain']} / {res['partner_table']} / {res['partner_column']} -> Base: {res['mapped_base_table']} / {res['mapped_base_column']} (Score: {res['confidence_score']:.2f})")