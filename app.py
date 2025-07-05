import streamlit as st
from backend import call_gemini

st.set_page_config(page_title="SQL Query Generator")

st.markdown("""
<style>
/* Wrap code blocks instead of scrolling */
code {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Gemini-Powered SQL Query Generator")
st.markdown("""
This app takes:
- 🟨 A user prompt (what you want to extract)
- 🟩 SQL context (DDL)
- 🟦 SQL sample data (INSERTs)

And generates a query using **Gemini Pro** LLM.
""")

sql_prompt = st.text_area("🟡 SQL Prompt", placeholder="e.g., What is the total number of hospital beds in each state?")
sql_context = st.text_area("🟢 SQL Context (DDL)", placeholder="CREATE TABLE Beds...")
sql_sample_data = st.text_area("🔵 SQL Sample Data (INSERTs)", placeholder="INSERT INTO Beds...")

if st.button("🚀 Generate SQL Query"):
    if not all([sql_prompt, sql_context, sql_sample_data]):
        st.warning("Please fill in all fields.")
    else:
        with st.spinner("Calling Gemini..."):
            result = call_gemini(sql_prompt, sql_context, sql_sample_data)
            st.code(result, language="sql")
