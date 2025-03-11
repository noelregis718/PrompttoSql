from model_utils import generate_sql
import streamlit as st

st.title("SQL Query Generator")

instruction = st.text_area("Enter your query:", placeholder="Write a SQL query to find all endangered species...")
context = st.text_area("Enter database schema:", placeholder="CREATE TABLE endangered_species...")

if st.button("Generate SQL"):
    if instruction and context:
        with st.spinner("Generating SQL..."):
            result = generate_sql(instruction, context)
            
            # Extract just the SQL part
            if "### Response:" in result:
                sql_part = result.split("### Response:")[1].split("### Explanation:")[0].strip()
                explanation = result.split("### Explanation:")[1].strip() if "### Explanation:" in result else ""
                
                st.subheader("Generated SQL:")
                st.code(sql_part, language="sql")
                
                st.subheader("Explanation:")
                st.write(explanation)
            else:
                st.write(result)
    else:
        st.error("Please provide both a query and database schema.")
