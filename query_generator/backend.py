# backend.py

import google.generativeai as genai
import json
from query_generator.config import GEMINI_API_KEY
from query_generator.llm_prompt import sql_generation_prompt

def call_gemini(sql_prompt: str, sql_context: str, sql_sample_data: str) -> str:
    try:
        full_prompt = sql_generation_prompt(sql_prompt, sql_context, sql_sample_data)

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(full_prompt)

        # Clean LLM response and parse JSON
        return extract_sql_from_json(response.text.strip())

    except Exception as e:
        return f"❌ Error: {str(e)}"


def extract_sql_from_json(llm_response: str) -> str:
    try:
        # Extract JSON from markdown block if enclosed in ```json
        if "```json" in llm_response:
            json_part = llm_response.split("```json")[1].split("```")[0].strip()
        else:
            json_part = llm_response

        parsed = json.loads(json_part)
        
        if isinstance(parsed, list) and "query" in parsed[0]:
            return parsed[0]["query"]
        else:
            return "Unexpected JSON structure."

    except json.JSONDecodeError:
        return f"Failed to parse JSON:\n{llm_response}"
