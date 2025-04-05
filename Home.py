import streamlit as st

# Page config
st.set_page_config(
    page_title="MathSense | Visualize Math Concepts",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theme state
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

theme_choice = st.sidebar.radio("Choose Theme", ["Dark", "Light"])
st.session_state.theme = theme_choice

# Theme-aware CSS injection
def inject_theme_css(mode):
    if mode == "Dark":
        st.markdown("""
            <style>
                html, body, [class*="css"] {
                    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
                    color: white !important;
                }

                .stApp {
                    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
                }

                .title-text, .subtitle-text, .about-title, .about-name, .about-email,
                .about-us-title, .about-us-name, .about-us-email {
                    color: white !important;
                }

                .feature-box {
                    background-color: #ffffffdd;
                    color: #222222 !important;
                }

                .get-started-button {
                    background-color: #FFA923;
                    color: white !important;
                }

                .get-started-button:hover {
                    background-color: #0056b3;
                }
            </style>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
            <style>
                html, body, [class*="css"] {
                    background-color: #f4f4f4;
                    color: black !important;
                }

                .stApp {
                    background-color: #f4f4f4;
                }

                .title-text, .subtitle-text, .about-title, .about-name, .about-email,
                .about-us-title, .about-us-name, .about-us-email {
                    color: black !important;
                }

                .feature-box {
                    background-color: #ffffff;
                    color: #222222 !important;
                }

                .get-started-button {
                    background-color: #FFA923;
                    color: white !important;
                }

                .get-started-button:hover {
                    background-color: #0056b3;
                }
            </style>
        """, unsafe_allow_html=True)

inject_theme_css(st.session_state.theme)

# Custom CSS (font, layout, static colors)
def load_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&display=swap');

            html, body, [class*="css"] {
                font-family: 'Raleway', sans-serif;
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
                margin-bottom: 10px;
                text-align: center;
            }

            .subtitle-text {
                font-size: 1.3em;
                margin-bottom: 30px;
                text-align: center;
            }

            .feature-box {
                padding: 1.2rem;
                border-radius: 10px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                margin-bottom: 20px;
                width: 100%;
            }

            .get-started-wrapper {
                display: flex;
                justify-content: center;
                width: 100%;
            }

            .get-started-button {
                display: inline-block;
                padding: 1em 2em;
                font-size: 1.1em;
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

            .about-section {
                margin-top: 40px;
                padding: 1.2rem;
                border-radius: 10px;
                width: 100%;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
            }

            .about-section h4 {
                margin-bottom: 10px;
            }

            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ---------- Main Content ----------
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
    <div class="get-started-wrapper">
        <a href="/Visualize_Mathematical_Concepts" class="get-started-button">Get Started</a>
    </div>
""", unsafe_allow_html=True)

# ---------- About Us Section ----------
st.markdown("""
    <div style="margin-top: 250px; width: 100%;">
        <h2 class="about-us-title" style="text-align: center; margin-bottom: 30px;">About Us</h2>
        <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 80px;">
            <div style="text-align: center; min-width: 200px;">
                <h4 class="about-us-name">Anuraag Patil</h4>
                <p class="about-us-email">anuraagpatil10@gmail.com</p>
            </div>
            <div style="text-align: center; min-width: 200px;">
                <h4 class="about-us-name">Anshul Gupta</h4>
                <p class="about-us-email">guptaanushul.1410@gmail.com</p>
            </div>
            <div style="text-align: center; min-width: 200px;">
                <h4 class="about-us-name">Dhruva V</h4>
                <p class="about-us-email">dhruva.nagveni@gmail.com</p>
            </div>
            <div style="text-align: center; min-width: 200px;">
                <h4 class="about-us-name">Prayag Goyani</h4>
                <p class="about-us-email">prayagbgoyani@gmail.com</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)



