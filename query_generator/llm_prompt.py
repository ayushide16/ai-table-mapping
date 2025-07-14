def sql_generation_prompt(sql_prompt: str, sql_context: str, sql_sample_data: str) -> str:
    return f"""
    You are an expert SQL generator.

    ### Task:
    Generate a SQL query that answers the following question based on the SQL Prompt, SQL context, and SQL Sample Data. 
    The SQL Prompt is the requirement of the user.
    The context gives the table schema(s).
    The sample data gives you and insight into the kind of data in the table(s).

    SQL Prompt:
    {sql_prompt}

    SQL Context:
    {sql_context}

    SQL Sample Data:
    {sql_sample_data}

    Respond with only the SQL query that answers the prompt.

    ** THINGS TO REMEMBER: **
    1. Replace every aggregate function’s column with a meaningful ALIAS using the `AS` keyword (e.g., `SUM(volume) AS total_volume`).
    2. When performing a JOIN, explicitly state the join condition using: `table1.column_name = table2.column_name`,  
       AND include the **column(s) used in the JOIN condition** in the SELECT clause.
    3. Use `JOIN` instead of `INNER JOIN`.
    4. Use ALL CAPS for **SQL keywords** (e.g., `SELECT`, `FROM`, `JOIN`, `GROUP BY`, `ORDER BY`).
    5. ALWAYS end the SQL query with a **semicolon** (`;`).

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

    """.strip()
