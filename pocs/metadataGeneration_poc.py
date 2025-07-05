import re
import csv
import pandas as pd
from io import StringIO
from datasets import load_dataset

# Load dataset
ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")

# Prepare data structures
all_tables = {}
all_samples = {}
all_fks = {}

for entry in ds:
    domain = entry["domain"]
    sql = entry["sql_context"]
    queries = [q.strip() for q in sql.strip().split(';') if q.strip()]

    for q in queries:
        if q.upper().startswith("CREATE SCHEMA"):
            continue

        if q.upper().startswith("CREATE TABLE"):
            m = re.match(r"CREATE TABLE(?: IF NOT EXISTS)? ([\w\.]+)\s*\((.*)\)", q, re.IGNORECASE | re.DOTALL)
            if not m: continue
            full_table, raw_cols = m.groups()
            table = full_table.split('.')[-1]
            raw_cols = raw_cols.replace('\n',' ').strip()
            cols = re.split(r',(?![^()]*\))', raw_cols)
            all_tables.setdefault(table, []).append((cols, domain))

        elif q.upper().startswith("INSERT INTO"):
            m = re.match(r"INSERT INTO ([\w\.]+)\s*\(([^)]+)\)\s*VALUES\s*(.+)", q, re.IGNORECASE | re.DOTALL)
            if not m: continue
            full_table, col_str, val_str = m.groups()
            table = full_table.split('.')[-1]
            cols = [c.strip() for c in col_str.split(',')]
            rows = re.findall(r"\(([^)]+)\)", val_str)
            if not rows: continue
            row = rows[0]
            reader = csv.reader(StringIO(row), skipinitialspace=True)
            vals = [v.strip().strip("'") for v in next(reader)]
            all_samples.setdefault(table, {})[tuple(cols)] = dict(zip(cols, vals))

# Build metadata
rows = []
for table, entries in all_tables.items():
    for cols_list, domain in entries:
        fk_map = {}
        columns = []
        for col_def in cols_list:
            cd = col_def.strip()
            fk = re.match(r"FOREIGN KEY\s*\((\w+)\)\s+REFERENCES\s+([\w\.]+)", cd, re.IGNORECASE)
            if fk:
                colname, ref = fk.groups()
                fk_map[colname] = ref.split('.')[-1]
                continue
            parts = cd.split(None,2)
            if len(parts) < 2: continue
            cname, dt = parts[0], parts[1]
            constraint = parts[2] if len(parts)==3 else ""
            columns.append((cname, dt, constraint))

        sample = {}
        for cols_tuple, smap in all_samples.get(table, {}).items():
            sample = smap; break

        for cname, dt, constraint in columns:
            rows.append({
                "domain": domain,
                "table_name": table,
                "column_name": cname,
                "data_type": dt,
                "constraint": constraint or "None",
                "sample_data": sample.get(cname, "—"),
                "foreign_key": "Yes" if cname in fk_map else "No",
                "foreign_table": fk_map.get(cname, "—")
            })

# Create DataFrame
df = pd.DataFrame(rows)[[
    "domain","table_name","column_name","data_type",
    "constraint","sample_data","foreign_key","foreign_table"
]]

# Save
df.to_csv("table_metadata.csv", index=False)

print("Metadata created and saved.")
