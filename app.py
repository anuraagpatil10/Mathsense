import streamlit as st
from utils.clt import generate_clt_plot
from utils.lln import generate_lln_plot
from utils.pca import generate_pca_comparison_plot
from utils.mmse import generate_mmse_plot
# from utils.poly_reg import generate_polynomial_regression_plot
from utils.linear_regression import plot_linear_regression
from utils.bayes_theorem import generate_bayes_plot
from utils.gradient_descent import generate_gradient_descent_plot
from utils.logistic_regression import plot_logistic_regression
from utils.lasso_ridge_reg import generate_lasso_ridge_plot
from utils.gaussian_naive_bayes import generate_gaussian_nb_plots

st.set_page_config(page_title="MathSense", layout="centered")
st.title("MathSense: Visualize Core Math Concepts")

section = st.sidebar.selectbox("Choose a Concept", [
    "Central Limit Theorem",
    "Law of Large Numbers",
    "Principal Component Analysis",
    "Minimum Mean Square Error",
    "Linear Regression",
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
    st.header("Principal Component Analysis (PCA)")

    n_components = st.slider("Select number of components", min_value=1, max_value=3, value=2)
    fig = generate_pca_comparison_plot(n_components=n_components)
    st.pyplot(fig)

    st.info(f"""
    PCA reduces data dimensions by projecting it to directions of maximum variance.

    - **Original plot** (left) shows real 3D data
    - **Right plot** shows projection into {n_components}D space
    - **Variance retained**: Higher components = more information
    """)

# === Section: MMSE ===
elif section == "Minimum Mean Square Error":
    st.header("MMSE Estimation")
    noise_level = st.slider("Noise Level", 0.1, 2.0, 0.5, step=0.1)
    fig = generate_mmse_plot(noise_level)
    st.pyplot(fig)
    st.info("MMSE is used to clean noisy data in signals, finance, and ML regression.")

# Linear regression
elif section == "Linear Regression":
    st.header("Linear Regression")

    st.subheader("Model Settings")
    mode = st.radio("Choose Mode", ["Train with data", "Manual weights/bias"])

    test_x = st.slider("Test Point X", 0.0, 2.0, 1.0, step=0.1)

    if mode == "Train with data":
        fig = plot_linear_regression(manual=False, test_x=test_x)
    else:
        w = st.slider("Weight (w)", -10.0, 10.0, 1.0, step=0.1)
        b = st.slider("Bias (b)", -10.0, 10.0, 0.0, step=0.1)
        fig = plot_linear_regression(manual=True, w=w, b=b, test_x=test_x)

    st.pyplot(fig)

    st.info(r"""
    **Linear Regression** predicts a target value using:

    \[
    y = wx + b
    \]

    - Adjust `w` and `b` manually to understand model behavior
    - Move the test point and watch the predicted output change!
    """)

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

# Logistic regression
elif section == "Logistic Regression":
    st.header("Logistic Regression")

    st.subheader("Model Settings")

    mode = st.radio("Choose Mode", ["Train with data", "Manual weights/bias"])

    test_x = st.slider("Test Point X", -5.0, 5.0, 0.0, step=0.1)
    test_y = st.slider("Test Point Y", -5.0, 5.0, 0.0, step=0.1)

    if mode == "Train with data":
        C = st.slider("Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
        fig = plot_logistic_regression(C=C, manual=False, test_point=(test_x, test_y))
    else:
        w0 = st.slider("Weight w0 (X1)", -5.0, 5.0, 1.0, step=0.1)
        w1 = st.slider("Weight w1 (X2)", -5.0, 5.0, 1.0, step=0.1)
        bias = st.slider("Bias", -5.0, 5.0, 0.0, step=0.1)
        fig = plot_logistic_regression(manual=True, w0=w0, w1=w1, bias=bias, test_point=(test_x, test_y))

    st.pyplot(fig)

    st.info("""
    Logistic regression estimates the probability that a point belongs to a class using the sigmoid function:

    \n$P(y=1|x) = \\frac{1}{1 + e^{-(w^T x + b)}}$

    - Adjust weights and bias to understand their effect
    - Try dragging the test point to see predicted probability!
    """)

elif section == "Lasso & Ridge Regression":
    st.header("Lasso & Ridge Regression")
    regression_type = st.radio("Select Regression Type", ["Lasso", "Ridge"])
    alpha = st.slider("Regularization Strength (α)", 0.01, 10.0, 1.0)
    fig = generate_lasso_ridge_plot(regression_type, alpha)
    st.pyplot(fig)
    st.info("Lasso (L1) helps with feature selection by shrinking coefficients to zero, while Ridge (L2) reduces overfitting without removing features.")

elif section == "Gaussian Naive Bayes":
    st.header("Gaussian Naive Bayes Classifier")

    correlation = st.slider("Adjust feature correlation", min_value=-0.9, max_value=0.9, value=0.0, step=0.1)
    fig1, fig2, test_points, probs = generate_gaussian_nb_plots(correlation=correlation)

    st.subheader("Decision Boundary")
    st.pyplot(fig1)

    st.subheader("Feature-wise Gaussian Distributions")
    st.pyplot(fig2)

    st.subheader("Posterior Probabilities for Sample Points")
    for i, (point, prob) in enumerate(zip(test_points, probs)):
        st.write(f"Test Point {i+1}: {point}")
        st.write(f"→ P(Class 0): {prob[0]:.2f}, P(Class 1): {prob[1]:.2f}")
    
elif section == "Other":
    st.header("Custom Visualization")
    custom_prompt = st.text_input("Describe the mathematical concept you want to visualize")

    if st.button("Generate Visualization"):
        if custom_prompt:
            with st.spinner("Generating visualization using Gemini..."):
                try:
                    from ai_backend import generate_visualization_code
                    from executor import execute_script

                    generated_code = generate_visualization_code(custom_prompt)
                    st.session_state.generated_code = generated_code
                    st.success("Visualization generated successfully!")
                except Exception as e:
                    st.error(f"Error while generating: {e}")

    # Display AI-generated visualization if code exists
    if st.session_state.get("generated_code"):
        from executor import execute_script
        execute_script(st.session_state.generated_code)
