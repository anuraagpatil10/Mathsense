import os
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
load_dotenv()
# Configure the API key
genai.configure(api_key = os.environ.get('API_KEY'))
# Initialize the Gemini Pro model
model = genai.GenerativeModel('gemini-1.5-pro')

def generate_visualization_code(user_prompt):
    """Generate Python visualization code using Gemini."""
    prompt = f"""
    Generate a Python script using Streamlit and Matplotlib to visualize the concept: {user_prompt}.
    Use interactive sliders if appropriate, and ensure the chart updates when sliders are used.
    Only return the code, no explanations or markdown.
    """
    response = model.generate_content(prompt)
    
    # Extract plain code (no markdown formatting)
    return response.text.strip().replace("```python", "").replace("```", "")
