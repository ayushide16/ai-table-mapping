from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import sys

# Add the backend directory to the Python path to allow importing rag_model
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
# Simpler: just ensure your `backend` directory is a Python package or directly import.
# For this structure, a direct relative import will work if run from `backend` directory.

try:
    # Changed from .rag_model to rag_model, and .config to config
    from rag_model import RAGSchemaMapper
    from query_generator.config import GEMINI_API_KEY
except ImportError as e:
    # Keep the error message for development, but it should now be less likely
    print(f"Error importing modules: {e}")
    print("A direct import failed. Ensure rag_model.py and config.py are in the same directory as app.py.")
    sys.exit(1)


# Initialize FastAPI app
app = FastAPI(
    title="AI Schema Mapping Service",
    description="An AI-powered service to map partner data columns to internal schema columns using RAG with Gemini.",
    version="1.0.0"
)

# Mount static files (CSS, JS)
# Assumes 'frontend' directory is one level up from 'backend'
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Configure Jinja2 templates (for serving index.html)
# Assumes 'frontend' directory is one level up from 'backend'
templates = Jinja2Templates(directory="../frontend")

# --- Initialize RAGSchemaMapper (loaded once when app starts) ---
# Ensure table_metadata.csv is in the root directory of the project
# (one level up from 'backend' directory).
METADATA_CSV_PATH = "../table_metadata.csv"
schema_mapper: RAGSchemaMapper # Declare type hint for clarity
try:
    schema_mapper = RAGSchemaMapper(metadata_path=METADATA_CSV_PATH, gemini_api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize RAGSchemaMapper on startup: {e}")
    print("This means the application cannot function. Please check the metadata file path and Gemini API key.")
    sys.exit(1) # Exit if core model loading fails

# Pydantic model for validating the request body for the /map-column endpoint
class PartnerColumnRequest(BaseModel):
    partner_domain_input: str
    partner_table_name: str
    partner_column_name: str
    partner_data_type: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main HTML page of the application.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/map-column")
async def map_column_endpoint(request: PartnerColumnRequest):
    """
    API endpoint to map a partner column to a base schema column.
    Accepts a JSON payload with partner column details and returns mapping results.
    """
    print(f"Received mapping request for: Domain='{request.partner_domain_input}', Table='{request.partner_table_name}', Column='{request.partner_column_name}'")

    try:
        mapping_result = schema_mapper.map_partner_column(
            partner_domain_input=request.partner_domain_input,
            partner_table_name=request.partner_table_name,
            partner_column_name=request.partner_column_name,
            partner_data_type=request.partner_data_type
        )
        return mapping_result
    except Exception as e:
        print(f"Error processing mapping request: {e}")
        # Return a more detailed error to the frontend if needed
        return {"error": str(e), "message": "An internal server error occurred during mapping.", "status": "backend_error"}

# You can run this backend using Uvicorn:
# 1. Make sure you are in the 'backend' directory.
# 2. Ensure your GOOGLE_API_KEY environment variable is set or in a .env file.
# 3. Run: uvicorn app:app --reload --port 8000
#    --reload is useful for development (reloads on code changes).
#    --port 8000 specifies the port.