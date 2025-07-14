import streamlit as st
import requests
import json # For pretty printing JSON

# --- Configuration ---
# URL of your FastAPI backend.
# If running locally, it's usually http://localhost:8000
# If your backend is deployed elsewhere, change this URL.
BACKEND_URL = "http://127.0.0.1:8000" # Or "http://localhost:8000"

st.set_page_config(
    page_title="AI Schema Mapper",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 AI Schema Mapper")
st.markdown("Map partner data columns to your internal schema using RAG and LLMs.")

st.header("Partner Column Details")

# Input fields for partner column metadata
partner_domain = st.text_input("Domain (e.g., forestry, defense industry, marine biology)", value="forestry")
partner_table = st.text_input("Table Name (from partner dataset)", value="forest_workers")
partner_column = st.text_input("Column Name (from partner dataset)", value="full_name")
partner_data_type = st.text_input("Data Type (from partner dataset)", value="TEXT")

# Button to trigger mapping
if st.button("Map Column"):
    # Basic input validation
    if not all([partner_domain, partner_table, partner_column, partner_data_type]):
        st.error("Please fill in all fields.")
    else:
        # Prepare data for the API request
        request_data = {
            "partner_domain_input": partner_domain,
            "partner_table_name": partner_table,
            "partner_column_name": partner_column,
            "partner_data_type": partner_data_type
        }

        st.info("Mapping in progress... Please wait (LLM call can take a few seconds).")

        try:
            # Make POST request to the FastAPI backend
            response = requests.post(f"{BACKEND_URL}/map-column", json=request_data)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            result = response.json()

            st.subheader("Mapping Result")

            if result.get("mapped_base_column"):
                st.success("Match Found!")
                st.write(f"**Partner Column:** {result.get('partner_domain')} / {result.get('partner_table')} / {result.get('partner_column')} ({result.get('partner_data_type')})")
                st.write(f"**Mapped Base Column:** {result.get('mapped_base_domain')} / {result.get('mapped_base_table')} / {result.get('mapped_base_column')} ({result.get('mapped_base_data_type')})")
                st.write(f"**FAISS Confidence Score:** {result.get('confidence_score', 0.0):.2f}")
                st.write(f"**LLM Explanation:** {result.get('llm_explanation', 'No explanation provided.')}")
            else:
                st.warning("No direct match found or an issue occurred.")
                st.write(f"**Partner Column:** {result.get('partner_domain')} / {result.get('partner_table')} / {result.get('partner_column')} ({result.get('partner_data_type')})")
                st.write(f"**Reason:** {result.get('llm_explanation', result.get('message', 'Unknown reason.'))}")

            st.markdown("---")
            st.subheader("Raw Backend Response (for debugging)")
            st.json(result) # Display the full JSON response for debugging

        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to the backend server at {BACKEND_URL}. Please ensure the FastAPI backend is running.")
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}. Please check backend logs.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")