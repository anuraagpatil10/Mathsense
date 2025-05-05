# MathSense - AI-Powered Mathematical Visualization App

**MathSense** is a dynamic Streamlit-based web application that helps users visualize mathematical and statistical concepts using AI-generated Python scripts. Whether you're a student, educator, or data science enthusiast, MathSense simplifies complex ideas through interactive visualizations — powered by Gemini AI and rendered instantly using Matplotlib, Seaborn, and more.

---

## Features

- **AI-Powered Code Generation**: Describe any concept like "Central Limit Theorem" or "KMeans Clustering" and let AI write the Python code for visualization.
- **Live Plot Rendering**: Visualize concepts directly using Python libraries such as Matplotlib and Seaborn.
- **Dynamic Interactivity**: Streamlit UI elements (sliders, dropdowns, etc.) allow users to control parameters in real time.
- **Dark/Light Mode Toggle**: Switch between elegant light and dark themes for optimal readability.
- **Error Handling**: User-friendly error messages if the AI can't generate meaningful visualizations.

---

## Use Cases

- Visualizing probability distributions, regression, clustering, etc.
- Learning or teaching data science, statistics, or applied math through visuals.

---

## Tech Stack

- **Frontend/UI**: Streamlit
- **AI Backend**: Gemini 1.5 Pro API
- **Visualization**: Matplotlib, Seaborn, NumPy, Scikit-learn
- **Environment Management**: Python + Virtualenv

---

## Installation & Running Locally

1. **Clone the repository**
```bash
git clone https://github.com/anuraagpatil10/mathsense.git
cd mathsense
```
2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up your Gemini API Key Create a .env file in the root directory and add your key:**
   ```bash
   API_KEY=your_gemini_api_key_here
   ```
5. **Run the app**
   ```bash
   streamlit run Home.py
   ```

