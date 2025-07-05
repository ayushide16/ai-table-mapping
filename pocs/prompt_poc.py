import duckdb
import sqlglot
import re
import json
import time
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import google.generativeai as genai




# -------------------------------
# 1. Setup Gemini + Dataset
# -------------------------------

# Replace with relevant Gemini API key
genai.configure(api_key="")
model = genai.GenerativeModel("gemini-2.0-flash")

# Load the Hugging Face dataset
ds = load_dataset("gretelai/synthetic_text_to_sql")
df = ds["train"].to_pandas()




# -------------------------------
# 2. Helper Functions
# -------------------------------

def normalize_sql(sql: str) -> str:
    try:
        return sqlglot.transpile(sql, read="sqlite", pretty=True)[0].strip().lower()
    except:
        return sql.strip().lower()

def execute_sql(sql_context: str, sql_query: str):
    try:
        con = duckdb.connect()
        con.sql(sql_context)
        result = con.sql(sql_query).fetchall()
        con.close()
        return result
    except Exception as e:
        return str(e)

def json_to_dict(json_string):
    # Remove the markdown code block delimiters
    if json_string.startswith('```json\n') and json_string.endswith('\n```'):
        json_string = json_string[8:-4]

    # Strip any leading or trailing whitespace
    json_string = json_string.strip()

    try:
        # Convert JSON string to dictionary
        #import pdb
        #pdb.set_trace()
        dictionary = json.loads(json_string)

        return dictionary
    except json.JSONDecodeError as e:
        print("Invalid JSON format:", e)
        return None

def build_prompt(prompt_question, context, task_description, complexity, description):
    return f"""
    You are a SQL generation model.
    
    ### Task:
    Generate a SQL query that answers the following question based on the SQL context, SQL task type description and SQL intent. 
    The context gives the table schema and sample data. The intent gives a heads-up regarding the task. 
    Remember to include identifiers used to process the query, i.e. PRIMARY KEY, FOREIGN KEY, etc.
    
    ### Question:
    {prompt_question}
    
    ### SQL Context:
    {context}
    
    ### SQL Task Type Description:
    {task_description}
    
    ### SQL Intent:
    {complexity} - {description}

    ** THINGS TO REMEMBER: **
    1. Replace every aggregate function’s column with a meaningful ALIAS using the `AS` keyword (e.g., `SUM(volume) AS total_volume`).
    2. When performing a JOIN, explicitly state the join condition using: `table1.column_name = table2.column_name`,  
       AND include the **column(s) used in the JOIN condition** in the SELECT clause.
    3. Use `JOIN` instead of `INNER JOIN`.
    4. Do NOT use **aliases** for table names (e.g., avoid `table_name AS t` or `t.column`).
    5. Use ALL CAPS for **SQL keywords** (e.g., `SELECT`, `FROM`, `JOIN`, `GROUP BY`, `ORDER BY`).
    6. ALWAYS end the SQL query with a **semicolon** (`;`).

    Follow the following order to generate the SQL query and include all components required
    "SELECT column_name, ...
    FROM table_name, ...
    WHERE condition_1, condition_2, ...
    GROUP BY column_name, ...
    HAVING condition
    ORDER BY column_name, ...
    LIMIT ...
    OFFSET ...;"
    
    **Format for Output to be followed strictly**:
    Return your answer in this exact JSON format:
    
    ```json
    [
        {{
            "query": "<YOUR_SQL_QUERY_HERE>"
        }}
    ]
    ```
    """




# -------------------------------
# 3. Evaluation Loop
# -------------------------------

results = []

for i, row in tqdm(df.iloc[10:20].iterrows(), total=10):
    sql_prompt = row["sql_prompt"]
    sql_context = row["sql_context"]
    ground_truth_sql = row["sql"]

    # --- Call Gemini ---
    try:
        prompt_text = build_prompt(
            sql_prompt,
            sql_context,
            row["sql_task_type_description"],
            row["sql_complexity"],
            row["sql_complexity_description"]
        )
        response = model.generate_content(prompt_text)
        response_json = json_to_dict(response.text)
        print(response_json)
        # generated_sql = response_json.get("query", "").strip()
        generated_sql = response_json[0]['query']

        # Optional: validate basic structure
        if not generated_sql.lower().startswith(("select", "insert", "update", "delete", "with")):
            raise ValueError("Returned SQL does not look valid.")

    except Exception as e:
        generated_sql = f"-- Gemini JSON parse failed: {str(e)}"

    # --- Evaluation Metrics ---
    try:
        exact_match = generated_sql.strip().lower() == ground_truth_sql.strip().lower()
        structural_match = normalize_sql(generated_sql) == normalize_sql(ground_truth_sql)
        execution_match = execute_sql(sql_context, generated_sql) == execute_sql(sql_context, ground_truth_sql)
    except:
        exact_match = structural_match = execution_match = False

    results.append({
        "index": i,
        "generated_sql": generated_sql,
        "ground_truth_sql": ground_truth_sql,
        "exact_match": exact_match,
        "structural_match": structural_match,
        "execution_match": execution_match
    })

    # Optional: Sleep to avoid Gemini rate limit
    time.sleep(1.5)



# -------------------------------
# 4. Results Summary
# -------------------------------

results_df = pd.DataFrame(results)

accuracy_summary = {
    "Exact Match Accuracy": round(results_df["exact_match"].mean() * 100, 2),
    "Structural Match Accuracy": round(results_df["structural_match"].mean() * 100, 2),
    "Execution Match Accuracy": round(results_df["execution_match"].mean() * 100, 2)
}

print("\nSample Results:\n", results_df.head())
print("\nAccuracy Summary:")
for metric, score in accuracy_summary.items():
    print(f"{metric}: {score}%")

# Optional: Save to CSV
# results_df.to_csv("gemini_sql_eval_results.json.csv", index=False)
