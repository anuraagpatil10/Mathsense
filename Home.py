import streamlit as st

# Page config
st.set_page_config(
    page_title="MathSense | Visualize Math Concepts",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Custom CSS ----------
def load_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&display=swap');

            html, body, [class*="css"]  {
                font-family: 'Raleway', sans-serif;
                background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
                color: #ffffff;
            }

            .main-container {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                
                max-width: 1000px;
                margin: 0 auto;
                padding: 2rem;
            }

            .title-text {
                font-size: 3em;
                font-weight: bold;
                color: #ffffff;
                margin-bottom: 10px;
                text-align: center;
            }

            .subtitle-text {
                font-size: 1.3em;
                color: #d0d0d0;
                margin-bottom: 30px;
                text-align: center;
            }

            .feature-box {
                background-color: #ffffffdd;
                color: #222222;
                padding: 1.2rem;
                border-radius: 10px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                margin-bottom: 20px;
                width: 100%;
            }

            .get-started-button {
                display: inline-block;
                padding: 1em 2em;
                font-size: 1.1em;
                color: white;
                background-color: #FFA923;
                border-radius: 8px;
                text-decoration: none;
                transition: all 0.3s ease;
                font-weight: bold;
                margin-top: 20px;
            }

            .get-started-button:hover {
                background-color: #0056b3;
                transform: scale(1.05);
            }

            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ---------- Center-Aligned Content ----------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown('<div class="title-text">Welcome to MathSense</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">A visual learning toolkit for mastering mathematical and statistical methods through interaction.</div>', unsafe_allow_html=True)

st.markdown("""
    <div class="feature-box">
    <h4>Key Modules</h4>
    <ul>
        <li>Central Limit Theorem</li>
        <li>Law of Large Numbers</li>
        <li>PCA & MMSE</li>
        <li>Gradient Descent</li>
        <li>Regression (Linear, Lasso, Ridge, Logistic)</li>
        <li>Gaussian Naive Bayes</li>
        <li>Neural Networks (with full visualizations!)</li>
    </ul>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="feature-box">
    <h4>Why MathSense?</h4>
    MathSense helps college students grasp abstract mathematical concepts with ease through interactive simulations and real-time visualization of formulas in action.
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <a href="/Visualize_Mathematical_Concepts" class="get-started-button">🚀 Get Started</a>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)




# import streamlit as st
# import base64
#
# # Page config
# st.set_page_config(
#     page_title="MathSense | Visualize Math Concepts",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )
#
# # ---------- Custom CSS ----------
# def load_custom_css():
#     st.markdown("""
#         <style>
#             @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&display=swap');
#
#             html, body, [class*="css"]  {
#                 font-family: 'Raleway', sans-serif;
#                 background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
#                 color: #ffffff;
#             }
#
#             .title-text {
#                 font-size: 3em;
#                 font-weight: bold;
#                 color: #ffffff;
#                 margin-bottom: 10px;
#             }
#
#             .subtitle-text {
#                 font-size: 1.3em;
#                 color: #d0d0d0;
#                 margin-bottom: 25px;
#             }
#
#             .feature-box {
#                 background-color: #ffffffdd;
#                 color: #222222;
#                 padding: 1.2rem;
#                 border-radius: 10px;
#                 box-shadow: 0 0 15px rgba(0,0,0,0.2);
#                 margin-bottom: 20px;
#             }
#
#             .get-started-button {
#                 display: inline-block;
#                 padding: 1em 2em;
#                 font-size: 1.1em;
#                 color: white;
#                 background-color:#FFA923;
#                 border-radius: 8px;
#                 text-decoration: none;
#                 transition: all 0.3s ease;
#                 font-weight: bold;
#             }
#
#             .get-started-button:hover {
#                 background-color: #0056b3;
#                 transform: scale(1.05);
#             }
#
#             footer {visibility: hidden;}
#         </style>
#     """, unsafe_allow_html=True)
#
# load_custom_css()
#
# # ---------- Page Content ----------
# st.markdown('<div class="title-text">Welcome to MathSense</div>', unsafe_allow_html=True)
# st.markdown('<div class="subtitle-text">A visual learning toolkit for mastering mathematical and statistical methods through interaction.</div>', unsafe_allow_html=True)
#
# # Two-column layout
# col1, col2 = st.columns([1.5, 1])
#
# with col1:
#     st.markdown("""
#         <div class="feature-box">
#         <h4>Key Modules</h4>
#         <ul>
#             <li>Central Limit Theorem</li>
#             <li>Law of Large Numbers</li>
#             <li>PCA & MMSE</li>
#             <li>Gradient Descent</li>
#             <li>Regression (Linear, Lasso, Ridge, Logistic)</li>
#             <li>Gaussian Naive Bayes</li>
#             <li>Neural Networks (with full visualizations!)</li>
#         </ul>
#         </div>
#     """, unsafe_allow_html=True)
#
#     st.markdown("""
#         <div class="feature-box">
#         <h4>Why MathSense?</h4>
#         MathSense helps college students grasp abstract mathematical concepts with ease through interactive simulations and real-time visualization of formulas in action.
#         </div>
#     """, unsafe_allow_html=True)
#
#     st.markdown("""
#         <a href="/Visualize_Mathematical_Concepts" class="get-started-button">Get Started</a>
#     """, unsafe_allow_html=True)
#
# with col2:
#     st.image("https://i.imgur.com/nHvmEx3.png", caption="Math through visuals", use_column_width=True)


#FFA923

# import streamlit as st
# import base64
#
# # Page config
# st.set_page_config(
#     page_title="MathSense | Visualize Math Concepts",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )
#
# # ---------- Custom CSS ----------
# def load_custom_css():
#     st.markdown("""
#         <style>
#             @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&display=swap');
#
#             html, body, [class*="css"]  {
#                 font-family: 'Raleway', sans-serif;
#                 background: linear-gradient(to right, #e0f7fa, #fce4ec);
#             }
#
#             .title-text {
#                 font-size: 3em;
#                 font-weight: bold;
#                 color: #2c3e50;
#                 margin-bottom: 10px;
#             }
#
#             .subtitle-text {
#                 font-size: 1.3em;
#                 color: #34495e;
#                 margin-bottom: 25px;
#             }
#
#             .feature-box {
#                 background-color: #ffffffcc;
#                 padding: 1.2rem;
#                 border-radius: 10px;
#                 box-shadow: 0 0 15px rgba(0,0,0,0.1);
#                 margin-bottom: 20px;
#             }
#
#             .get-started-button {
#                 display: inline-block;
#                 padding: 1em 2em;
#                 font-size: 1.1em;
#                 color: white;
#                 background-color: #007BFF;
#                 border-radius: 8px;
#                 text-decoration: none;
#                 transition: all 0.3s ease;
#             }
#
#             .get-started-button:hover {
#                 background-color: #0056b3;
#                 transform: scale(1.05);
#             }
#
#             footer {visibility: hidden;}
#         </style>
#     """, unsafe_allow_html=True)
#
# load_custom_css()
#
# # ---------- Page Content ----------
# st.markdown('<div class="title-text">📊 Welcome to MathSense</div>', unsafe_allow_html=True)
# st.markdown('<div class="subtitle-text">A visual learning toolkit for mastering mathematical and statistical methods through interaction.</div>', unsafe_allow_html=True)
#
# # Two-column layout
# col1, col2 = st.columns([1.5, 1])
#
# with col1:
#     st.markdown("""
#         <div class="feature-box">
#         <h4>🎯 Key Modules</h4>
#         - Central Limit Theorem
#         - Law of Large Numbers
#         - PCA & MMSE
#         - Gradient Descent
#         - Regression (Linear, Lasso, Ridge, Logistic)
#         - Gaussian Naive Bayes
#         - Neural Networks (with full visualizations!)
#         </div>
#     """, unsafe_allow_html=True)
#
#     st.markdown("""
#         <div class="feature-box">
#         <h4>🧠 Why MathSense?</h4>
#         MathSense helps college students grasp abstract mathematical concepts with ease through interactive simulations and real-time visualization of formulas in action.
#         </div>
#     """, unsafe_allow_html=True)
#
#     st.markdown("""
#         <a href="/Visualize_Mathematical_Concepts" class="get-started-button">🚀 Get Started</a>
#     """, unsafe_allow_html=True)

# with col2:
#     st.image("https://i.imgur.com/nHvmEx3.png", caption="Math through visuals", use_column_width=True)




# import streamlit as st
# from PIL import Image
#
# st.set_page_config(page_title="MathSense", layout="centered")
#
# # Optional logo
# # st.image("assets/logo.png", width=120)  # if you have a logo
#
# st.title("Welcome to MathSense")
# st.markdown("""
# MathSense is your interactive toolkit to **visualize and understand core mathematical methods** used in data science, machine learning, and statistics.
#
# Whether you're trying to wrap your head around the **Central Limit Theorem**, simulate **Gradient Descent**, or tweak a **Neural Network**, we've got you covered.
# """)
#
# st.markdown("### Key Concepts You Can Explore")
# st.markdown("""
# - Central Limit Theorem
# - Law of Large Numbers
# - MMSE (Minimum Mean Square Error)
# - PCA (Principal Component Analysis)
# - Linear, Logistic, Lasso, Ridge Regression
# - Gaussian Naive Bayes
# - Neural Networks and more...
# """)
#
# st.markdown("---")
#
# # Call to action button
# if st.button("Get Started"):
#     st.switch_page("pages/Visualize_Mathematical_Concepts.py")
#   # works with multipage setup
