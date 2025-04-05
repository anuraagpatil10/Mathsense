import os
import google.generativeai as genai
from dotenv import load_dotenv
import io

load_dotenv()
genai.configure(api_key=os.environ.get('API_KEY'))
model = genai.GenerativeModel('gemini-1.5-pro')

# Step 1: Strictly validate if concept is visualizable
def is_visualizable_math_concept(user_prompt):
    check_prompt = f"""
    Consider the user input: "{user_prompt}"

    Is this a valid mathematical or statistical concept that can be meaningfully visualized using Python libraries 
    like Matplotlib or Seaborn — using plots, charts, or visual simulations?

    Only respond with one word: Yes or No.
    Do NOT provide explanation or context.
    """
    response = model.generate_content(check_prompt).text.strip().lower()
    return response.startswith("yes")

# Step 2: Generate visualization code if valid
def generate_visualization_code(user_prompt, csv_sample=None):
    if not is_visualizable_math_concept(user_prompt):
        return (
            "# ❌ The entered topic doesn't appear to be a visualizable mathematical concept.\n"
            "# Please try a topic that can be represented with a chart, graph, or simulation."
        )
    
    gen_prompt = f"""
    Generate a Python script using Streamlit and Matplotlib to visualize the concept: {user_prompt}.
    Use sliders if applicable, and base it on the following sample CSV data:\n{csv_sample if csv_sample else "No data"}.
    Only return the code, no markdown, no explanations.
    """
    response = model.generate_content(gen_prompt)
    return response.text.strip().replace("```python", "").replace("```", "")
