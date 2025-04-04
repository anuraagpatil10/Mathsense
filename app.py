import streamlit as st
from utils.clt import generate_clt_plot
from utils.lln import generate_lln_plot
from utils.pca import generate_pca_plot
from utils.mmse import generate_mmse_plot
from utils.poly_reg import generate_polynomial_regression_plot
from utils.bayes_theorem import generate_bayes_plot
from utils.gradient_descent import generate_gradient_descent_plot
from utils.logistic_regression import generate_logistic_regression_plot
from utils.lasso_ridge_reg import generate_lasso_ridge_plot
from utils.gaussian_naive_bayes import generate_gaussian_nb_plot

st.set_page_config(page_title="MathSense", layout="centered")
st.title("MathSense: Visualize Core Math Concepts")

section = st.sidebar.selectbox("Choose a Concept", [
    "Central Limit Theorem",
    "Law of Large Numbers",
    "Principal Component Analysis",
    "Minimum Mean Square Error",
    "Polynomial Regression",
    "Bayes’ Theorem",
    "Gradient Descent",
    "Logistic Regression",
    "Lasso & Ridge Regression",
    "Gaussian Naive Bayes",
    "Other"
])

st.sidebar.markdown("---")
st.sidebar.write("Designed for college students to gain intuition and confidence in mathematical concepts used in real-world careers.")

# === Section: CLT ===
if section == "Central Limit Theorem":
    st.header("Central Limit Theorem")
    distribution = st.selectbox("Population Distribution", ["Normal", "Uniform", "Exponential"])
    sample_size = st.slider("Sample Size (n)", 1, 100, 30)
    num_samples = st.slider("Number of Samples", 100, 5000, 1000)
    fig = generate_clt_plot(distribution, sample_size, num_samples)
    st.pyplot(fig)
    st.info("CLT is used in A/B testing, polling, and startup metrics!")

# === Section: LLN ===
elif section == "Law of Large Numbers":
    st.header("Law of Large Numbers")
    distribution = st.selectbox("Distribution", ["Normal", "Uniform", "Exponential"])
    trials = st.slider("Number of Trials", 100, 10000, 1000)
    fig = generate_lln_plot(distribution, trials)
    st.pyplot(fig)
    st.info("LLN explains why more data leads to stable ML results.")

# === Section: PCA ===
elif section == "Principal Component Analysis":
    st.header("Principal Component Analysis")
    n_samples = st.slider("Number of Samples", 100, 1000, 300)
    n_features = st.slider("Number of Features", 2, 10, 5)
    n_components = st.selectbox("Components to Visualize", [2, "All"])
    n_comp_val = n_features if n_components == "All" else 2
    fig = generate_pca_plot(n_samples, n_features, n_comp_val)
    st.pyplot(fig)
    st.info("PCA is used in image compression, finance, and ML feature reduction.")

# === Section: MMSE ===
elif section == "Minimum Mean Square Error":
    st.header("MMSE Estimation")
    noise_level = st.slider("Noise Level", 0.1, 2.0, 0.5, step=0.1)
    fig = generate_mmse_plot(noise_level)
    st.pyplot(fig)
    st.info("MMSE is used to clean noisy data in signals, finance, and ML regression.")

elif section == "Polynomial Regression":
    st.header("Polynomial Regression")
    degree = st.slider("Polynomial Degree", 1, 10, 3)
    noise_level = st.slider("Noise Level", 0.1, 2.0, 0.5)
    fig = generate_polynomial_regression_plot(degree, noise_level)
    st.pyplot(fig)
    st.info("Polynomial Regression is used in finance, weather forecasting, and data trend analysis.")

elif section == "Bayes’ Theorem":
    st.header("Bayes' Theorem Visualization")
    prior_A = st.slider("Prior Probability P(A)", 0.01, 1.0, 0.5)
    prob_B_given_A = st.slider("Likelihood P(B|A)", 0.01, 1.0, 0.7)
    prob_B_given_not_A = st.slider("Likelihood P(B|¬A)", 0.01, 1.0, 0.2)
    fig = generate_bayes_plot(prior_A, prob_B_given_A, prob_B_given_not_A)
    st.pyplot(fig)
    st.info("Bayes’ Theorem is used in spam filters, medical diagnosis, and probabilistic AI models.")

elif section == "Gradient Descent":
    st.header("Gradient Descent")
    learning_rate = st.slider("Learning Rate (α)", 0.001, 0.1, 0.01)
    epochs = st.slider("Number of Epochs", 10, 200, 50)
    fig = generate_gradient_descent_plot(learning_rate, epochs)
    st.pyplot(fig)
    st.info("Gradient Descent is how machine learning models minimize error and improve accuracy over time.")

elif section == "Logistic Regression":
    st.header("Logistic Regression")
    activation = st.radio("Choose Activation Function", ["sigmoid", "relu"])
    fig = generate_logistic_regression_plot(activation)
    st.pyplot(fig)
    st.info(
        "Logistic Regression with sigmoid is classic, but ReLU shows what happens with alternative activation in binary classification.")

elif section == "Lasso & Ridge Regression":
    st.header("Lasso & Ridge Regression")
    regression_type = st.radio("Select Regression Type", ["Lasso", "Ridge"])
    alpha = st.slider("Regularization Strength (α)", 0.01, 10.0, 1.0)
    fig = generate_lasso_ridge_plot(regression_type, alpha)
    st.pyplot(fig)
    st.info("Lasso (L1) helps with feature selection by shrinking coefficients to zero, while Ridge (L2) reduces overfitting without removing features.")

elif section == "Gaussian Naive Bayes":
    st.header("Gaussian Naive Bayes Classifier")
    fig = generate_gaussian_nb_plot()
    st.pyplot(fig)
    st.info("Gaussian Naive Bayes assumes features follow a normal distribution. It's simple, fast, and surprisingly effective for text, spam filtering, and medical data.")