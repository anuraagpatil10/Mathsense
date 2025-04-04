import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(X, y, lr=0.01, epochs=50):
    m = 0  # slope
    c = 0  # intercept
    n = len(X)

    m_history = []
    c_history = []
    loss_history = []

    for _ in range(epochs):
        y_pred = m * X + c
        error = y_pred - y
        loss = np.mean(error ** 2)

        # gradients
        dm = (2 / n) * np.dot(error, X)
        dc = (2 / n) * np.sum(error)

        # update
        m -= lr * dm
        c -= lr * dc

        m_history.append(m)
        c_history.append(c)
        loss_history.append(loss)

    return m_history, c_history, loss_history


def generate_gradient_descent_plot(learning_rate, epochs):
    # Generate synthetic linear data
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = 2.5 * X + 5 + np.random.randn(50)

    m_hist, c_hist, loss_hist = gradient_descent(X, y, lr=learning_rate, epochs=epochs)

    # Plot loss over epochs
    fig, ax = plt.subplots()
    ax.plot(range(epochs), loss_hist, marker='o')
    ax.set_title("Loss Reduction via Gradient Descent")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean Squared Error (Loss)")
    return fig
