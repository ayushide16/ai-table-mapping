import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import os
from typing import List, Dict, Optional

class RAGSchemaMapper:
    def __init__(self, metadata_path: str, gemini_api_key: str):
        self.metadata_path = metadata_path
        self.gemini_api_key = gemini_api_key
        self.base_metadata_df: Optional[pd.DataFrame] = None
        self.model: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.gemini_model: Optional[genai.GenerativeModel] = None

        self._load_models_and_data()

    def _load_models_and_data(self):
        """Loads base metadata, initializes embedding model, and builds FAISS index."""
        print("--- RAGSchemaMapper: Initializing... ---")
        try:
            # 1. Load Base Metadata
            self.base_metadata_df = pd.read_csv(self.metadata_path)
            if 'sql_context' in self.base_metadata_df.columns:
                self.base_metadata_df = self.base_metadata_df.drop(columns=['sql_context'])
            print(f"--- Base Metadata Loaded from {self.metadata_path} ---")
            print(self.base_metadata_df.head())

            # 2. Initialize the Embedding Model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("--- Sentence Transformer Model Loaded ---")

            # 3. Prepare Text for Embeddings (Chunking)
            self.base_metadata_df['embedding_text'] = self.base_metadata_df.apply(
                lambda row: f"Domain: {row['domain']}. Table: {row['table_name']}. Column: {row['column_name']}. Data Type: {row['data_type']}.",
                axis=1
            )
            print("--- Embedding texts prepared (data chunked per row) ---")

            # 4. Generate Embeddings for all Base Metadata
            print("--- Generating Embeddings for Base Metadata (This might take a moment)... ---")
            base_embeddings = self.model.encode(self.base_metadata_df['embedding_text'].tolist(), show_progress_bar=True)
            embedding_dimension = base_embeddings.shape[1]
            print(f"Generated {len(base_embeddings)} embeddings with dimension {embedding_dimension}")

            # 5. Create a FAISS Index
            self.faiss_index = faiss.IndexFlatIP(embedding_dimension)
            self.faiss_index.add(base_embeddings)
            print(f"--- FAISS Index Created and Populated with {self.faiss_index.ntotal} vectors ---")

            # Initialize Gemini Model
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                print("--- Gemini Model Initialized ---")
            else:
                print("WARNING: Gemini API Key not provided. LLM generation will not be available.")

        except FileNotFoundError:
            print(f"Error: '{self.metadata_path}' not found. Please make sure the file is in the correct directory.")
            raise
        except Exception as e:
            print(f"Error during RAGSchemaMapper initialization: {e}")
            raise

    def _are_data_types_compatible(self, type1: str, type2: str) -> bool:
        """Helper function for data type compatibility (as discussed previously)."""
        type1 = type1.lower().split('(')[0]
        type2 = type2.lower().split('(')[0]

        if type1 == type2:
            return True
        num_types = ["int", "real", "numeric", "decimal", "float", "double"]
        if any(n_type in type1 for n_type in num_types) and any(n_type in type2 for n_type in num_types):
            return True
        text_types = ["char", "text", "varchar", "string"]
        if any(t_type in type1 for t_type in text_types) and any(t_type in type2 for t_type in text_types):
            return True
        date_types = ["date", "time", "timestamp", "datetime"]
        if any(d_type in type1 for d_type in date_types) and any(d_type in type2 for d_type in date_types):
            return True
        return False

    def map_partner_column(
        self,
        partner_domain_input: str,
        partner_table_name: str,
        partner_column_name: str,
        partner_data_type: str,
        top_k_retrieval: int = 10,
        top_k_llm_consideration: int = 3,
        min_confidence_retrieval: float = 0.5
    ) -> Dict:
        """
        Maps a single partner column to the best matching base column using RAG and LLM.
        Returns a dictionary with mapping results and LLM explanation.
        """
        if self.base_metadata_df is None or self.model is None or self.faiss_index is None:
            return {"error": "RAGSchemaMapper not initialized. Please check server logs.", "message": "Backend initialization error."}

        print(f"\n--- Mapping Partner Column: {partner_domain_input} / {partner_table_name} / {partner_column_name} ({partner_data_type}) ---")

        # 1. Prepare Query Text (Handling user query)
        partner_query_text = f"Domain: {partner_domain_input}. Table: {partner_table_name}. Column: {partner_column_name}. Data Type: {partner_data_type}."
        print(f"Query Text for Embedding: '{partner_query_text}'")

        # 2. Generate Query Embedding
        query_embedding = self.model.encode([partner_query_text])

        # 3. Filter Base Metadata by User-Provided Domain (Crucial pre-filtering)
        domain_filtered_indices = self.base_metadata_df[
            self.base_metadata_df['domain'].str.lower() == partner_domain_input.lower()
        ].index.values

        if len(domain_filtered_indices) == 0:
            return {
                "partner_domain": partner_domain_input,
                "partner_table": partner_table_name,
                "partner_column": partner_column_name,
                "partner_data_type": partner_data_type,
                "mapped_base_domain": None,
                "mapped_base_table": None,
                "mapped_base_column": None,
                "mapped_base_data_type": None,
                "confidence_score": 0.0,
                "llm_explanation": f"No base metadata found for domain: '{partner_domain_input}'. Please ensure the domain is correct.",
                "llm_raw_response": "",
                "status": "no_domain_match"
            }

        # Perform similarity search on the full index (retrieving more than top_k_retrieval to account for filtering)
        distances, all_indices = self.faiss_index.search(query_embedding, top_k_retrieval * 2)

        # Filter results by domain and data type compatibility, and collect candidates for LLM
        retrieved_candidates_for_llm: List[Dict] = []
        print("\nInitial Retrieved Candidates (before LLM):")
        for i, idx in enumerate(all_indices[0]):
            # Check if the retrieved index is within the allowed domain AND meets minimum confidence
            if idx in domain_filtered_indices and distances[0][i] >= min_confidence_retrieval:
                base_row = self.base_metadata_df.iloc[idx]
                # And finally, check data type compatibility
                if self._are_data_types_compatible(partner_data_type, base_row['data_type']):
                    candidate = {
                        "base_idx": int(idx), # Ensure int for JSON serialization
                        "score": float(distances[0][i]), # Ensure float for JSON serialization
                        "base_domain": base_row['domain'],
                        "base_table_name": base_row['table_name'],
                        "base_column_name": base_row['column_name'],
                        "base_data_type": base_row['data_type']
                    }
                    retrieved_candidates_for_llm.append(candidate)
                    print(f"  Candidate {len(retrieved_candidates_for_llm)} (Score: {candidate['score']:.2f}): {candidate['base_domain']} / {candidate['base_table_name']} / {candidate['base_column_name']} ({candidate['base_data_type']})")
                    if len(retrieved_candidates_for_llm) >= top_k_llm_consideration:
                        break # Limit candidates sent to LLM

        if not retrieved_candidates_for_llm:
            return {
                "partner_domain": partner_domain_input,
                "partner_table": partner_table_name,
                "partner_column": partner_column_name,
                "partner_data_type": partner_data_type,
                "mapped_base_domain": None,
                "mapped_base_table": None,
                "mapped_base_column": None,
                "mapped_base_data_type": None,
                "confidence_score": 0.0,
                "llm_explanation": "No suitable candidates found after initial retrieval, domain, and data type filtering.",
                "llm_raw_response": "",
                "status": "no_candidates"
            }

        retrieved_candidates_for_llm.sort(key=lambda x: x['score'], reverse=True)

        # 4. Generate Response with LLM (Gemini)
        if not self.gemini_model:
            print("WARNING: Gemini model not initialized. Returning top FAISS match without LLM explanation.")
            best_faiss_match = retrieved_candidates_for_llm[0]
            return {
                "partner_domain": partner_domain_input,
                "partner_table": partner_table_name,
                "partner_column": partner_column_name,
                "partner_data_type": partner_data_type,
                "mapped_base_domain": best_faiss_match['base_domain'],
                "mapped_base_table": best_faiss_match['base_table_name'],
                "mapped_base_column": best_faiss_match['base_column_name'],
                "mapped_base_data_type": best_faiss_match['base_data_type'],
                "confidence_score": best_faiss_match['score'],
                "llm_explanation": "Gemini model not available. This is the top FAISS retrieval match.",
                "llm_raw_response": "N/A - Gemini not initialized",
                "status": "faiss_only"
            }

        prompt_parts = [
            f"You are an expert data schema mapping assistant. Your task is to identify the best match for a given partner data column from a list of potential base columns. You MUST output your final decision in a clear, concise format, stating the chosen base column, its table, and its domain, followed by a brief explanation. If no good match is found, state that.\n\n",
            f"**Partner Column to Map:**\n",
            f"Domain: {partner_domain_input}\n",
            f"Table: {partner_table_name}\n",
            f"Column Name: {partner_column_name}\n",
            f"Data Type: {partner_data_type}\n\n",
            f"**Potential Base Column Matches (from our internal schema):**\n"
        ]

        for i, candidate in enumerate(retrieved_candidates_for_llm):
            prompt_parts.append(
                f"{i+1}. Domain: {candidate['base_domain']}, Table: {candidate['base_table_name']}, Column: {candidate['base_column_name']}, Data Type: {candidate['base_data_type']} (Similarity Score: {candidate['score']:.2f})\n"
            )

        prompt_parts.append(
            f"\nBased on semantic meaning, data types, and context, which of the above 'Potential Base Column Matches' is the *best* match for the 'Partner Column to Map'? Please provide your answer in the format:\n"
            f"BEST MATCH: [Base Domain] / [Base Table Name] / [Base Column Name]\n"
            f"EXPLANATION: [Your concise reasoning]\n\n"
            f"If you believe none are a good match, respond with:\n"
            f"BEST MATCH: None\n"
            f"EXPLANATION: [Reason why no good match was found]"
        )

        full_prompt = "".join(prompt_parts)
        print("\n--- Sending Prompt to Gemini ---")

        llm_raw_response = ""
        try:
            response = self.gemini_model.generate_content(full_prompt)
            llm_raw_response = response.text
            print("\n--- Gemini's Raw Response ---")
            print(llm_raw_response)

            best_match_line = None
            explanation_line = None
            for line in llm_raw_response.split('\n'):
                if line.startswith("BEST MATCH:"):
                    best_match_line = line.replace("BEST MATCH:", "").strip()
                elif line.startswith("EXPLANATION:"):
                    explanation_line = line.replace("EXPLANATION:", "").strip()

            result = {
                "partner_domain": partner_domain_input,
                "partner_table": partner_table_name,
                "partner_column": partner_column_name,
                "partner_data_type": partner_data_type,
                "mapped_base_domain": None,
                "mapped_base_table": None,
                "mapped_base_column": None,
                "mapped_base_data_type": None,
                "confidence_score": 0.0,
                "llm_explanation": explanation_line if explanation_line else "LLM did not provide a clear explanation or specific match.",
                "llm_raw_response": llm_raw_response,
                "status": "llm_processed"
            }

            if best_match_line and best_match_line.lower() != "none":
                try:
                    parts = [p.strip() for p in best_match_line.split('/')]
                    if len(parts) == 3:
                        chosen_domain, chosen_table, chosen_column = parts
                        # Find the chosen candidate from our retrieved list to get its full details and score
                        final_match_candidate = next((c for c in retrieved_candidates_for_llm if
                                                       c['base_domain'].lower() == chosen_domain.lower() and
                                                       c['base_table_name'].lower() == chosen_table.lower() and
                                                       c['base_column_name'].lower() == chosen_column.lower()), None)
                        if final_match_candidate:
                            result.update({
                                "mapped_base_domain": final_match_candidate['base_domain'],
                                "mapped_base_table": final_match_candidate['base_table_name'],
                                "mapped_base_column": final_match_candidate['base_column_name'],
                                "mapped_base_data_type": final_match_candidate['base_data_type'],
                                "confidence_score": final_match_candidate['score'],
                                "status": "mapped_by_llm"
                            })
                        else:
                            print(f"Warning: LLM chose a column not found in initial candidates. Falling back to top FAISS match.")
                            top_faiss_candidate = retrieved_candidates_for_llm[0]
                            result.update({
                                "mapped_base_domain": top_faiss_candidate['base_domain'],
                                "mapped_base_table": top_faiss_candidate['base_table_name'],
                                "mapped_base_column": top_faiss_candidate['base_column_name'],
                                "mapped_base_data_type": top_faiss_candidate['base_data_type'],
                                "confidence_score": top_faiss_candidate['score'],
                                "llm_explanation": f"LLM chose a non-candidate. Fallback to top FAISS match. Reason: {result['llm_explanation']}",
                                "status": "llm_fallback_to_faiss"
                            })

                    else:
                        print(f"Warning: Unexpected format from LLM for BEST MATCH: '{best_match_line}'. Falling back to top FAISS match.")
                        top_faiss_candidate = retrieved_candidates_for_llm[0]
                        result.update({
                            "mapped_base_domain": top_faiss_candidate['base_domain'],
                            "mapped_base_table": top_faiss_candidate['base_table_name'],
                            "mapped_base_column": top_faiss_candidate['base_column_name'],
                            "mapped_base_data_type": top_faiss_candidate['base_data_type'],
                            "confidence_score": top_faiss_candidate['score'],
                            "llm_explanation": f"LLM output format error. Fallback to top FAISS match. Reason: {result['llm_explanation']}",
                            "status": "llm_format_error_fallback"
                        })

                except Exception as parse_e:
                    print(f"Error parsing LLM best match line: {parse_e}. Falling back to top FAISS match.")
                    top_faiss_candidate = retrieved_candidates_for_llm[0]
                    result.update({
                        "mapped_base_domain": top_faiss_candidate['base_domain'],
                        "mapped_base_table": top_faiss_candidate['base_table_name'],
                        "mapped_base_column": top_faiss_candidate['base_column_name'],
                        "mapped_base_data_type": top_faiss_candidate['base_data_type'],
                        "confidence_score": top_faiss_candidate['score'],
                        "llm_explanation": f"Error parsing LLM response. Fallback to top FAISS match. Reason: {result['llm_explanation']}",
                        "status": "llm_parse_error_fallback"
                    })
            else:
                # LLM explicitly said "None"
                result["llm_explanation"] = explanation_line if explanation_line else "LLM explicitly determined no good match from candidates."
                result["status"] = "llm_no_match"

            return result

        except genai.types.BlockedPromptException as e:
            print(f"Error: Gemini API call blocked. Reason: {e.response.prompt_feedback.block_reason.name}")
            return {
                "partner_domain": partner_domain_input,
                "partner_table": partner_table_name,
                "partner_column": partner_column_name,
                "partner_data_type": partner_data_type,
                "mapped_base_domain": None,
                "mapped_base_table": None,
                "mapped_base_column": None,
                "mapped_base_data_type": None,
                "confidence_score": 0.0,
                "llm_explanation": f"Gemini API call blocked: {e.response.prompt_feedback.block_reason.name}. Please refine your query.",
                "llm_raw_response": "",
                "status": "llm_blocked"
            }
        except Exception as e:
            print(f"An unexpected error occurred during Gemini API call: {e}")
            return {
                "partner_domain": partner_domain_input,
                "partner_table": partner_table_name,
                "partner_column": partner_column_name,
                "partner_data_type": partner_data_type,
                "mapped_base_domain": None,
                "mapped_base_table": None,
                "mapped_base_column": None,
                "mapped_base_data_type": None,
                "confidence_score": 0.0,
                "llm_explanation": f"An unexpected error occurred with the LLM service: {e}. Please check backend logs.",
                "llm_raw_response": "",
                "status": "llm_error"
            }