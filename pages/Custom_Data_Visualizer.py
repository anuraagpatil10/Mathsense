import streamlit as st
import pandas as pd
from ai_backend import generate_visualization_code
from executor import execute_script
import io
st.set_page_config(page_title="MathSense | Custom Data", layout="centered")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
user_prompt = st.text_input("Describe the mathematical concept to visualize (e.g., PCA, Regression, etc.)")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        if user_prompt:
            st.info("Generating visualization using AI...")
            code = generate_visualization_code(user_prompt, df.head().to_csv(index=False))
            output = execute_script(code)
            if output.get("error"):
                st.error(output["error"])
            elif output and "plot" in output:
                st.pyplot(output["plot"])
                if "warnings" in output:
                    st.warning(output["warnings"])
            else:
                st.warning("Something unexpected happened. Please try again.")

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Please upload a CSV file to begin.")
