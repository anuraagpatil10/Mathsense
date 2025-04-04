import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def generate_polynomial_regression_plot(degree, noise_level):
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_true = X**3 - 2*X**2 + X  # True underlying function
    y_noisy = y_true + np.random.normal(0, noise_level, X.shape)  # Add noise

    # Fit Polynomial Regression
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y_noisy)

    # Predict
    y_pred = model.predict(X_poly)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(X, y_noisy, color='gray', label="Noisy Data", alpha=0.5)
    ax.plot(X, y_true, label="True Function", linestyle="dashed", color="black")
    ax.plot(X, y_pred, label=f"Polynomial Regression (Degree {degree})", color="blue")
    ax.set_title("Polynomial Regression Curve Fitting")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    return fig
