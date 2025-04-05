import numpy as np
import matplotlib.pyplot as plt

def plot_gradient_descent(learning_rate=0.1, iterations=10, manual=False, w=1.0, b=0.0, test_x=1.0):
    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X[:, 0] + np.random.randn(100)

    m = len(X)

    if not manual:
        # Initialize
        w, b = 0.0, 0.0

        # Gradient Descent
        for _ in range(iterations):
            y_pred = w * X[:, 0] + b
            dw = (2/m) * np.dot(X[:, 0], (y_pred - y))
            db = (2/m) * np.sum(y_pred - y)
            w -= learning_rate * dw
            b -= learning_rate * db

    # Final prediction
    y_pred = w * X[:, 0] + b
    test_y = w * test_x + b

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', alpha=0.5, label='Data')
    ax.plot(X, y_pred, color='purple', linewidth=2, label='Gradient Descent Line')
    ax.plot(test_x, test_y, 'yo', markersize=10, label=f"Predicted: {test_y:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Gradient Descent (Linear Regression)")
    ax.legend()

    return fig


