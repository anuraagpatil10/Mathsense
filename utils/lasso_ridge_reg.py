import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def generate_lasso_ridge_plot(regression_type="Lasso", alpha=1.0):
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 3 * X.squeeze() + np.random.randn(100) * 2  # Linear relationship with noise

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose model
    if regression_type == "Lasso":
        model = Lasso(alpha=alpha)
    else:
        model = Ridge(alpha=alpha)

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute loss
    mse = mean_squared_error(y_test, y_pred)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(X_train, y_train, color="blue", label="Training Data", alpha=0.6)
    ax.scatter(X_test, y_test, color="red", label="Test Data", alpha=0.6)
    ax.plot(X_test, y_pred, color="black", label=f"{regression_type} Regression (α={alpha})")
    ax.set_title(f"{regression_type} Regression (MSE: {mse:.2f})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    return fig
