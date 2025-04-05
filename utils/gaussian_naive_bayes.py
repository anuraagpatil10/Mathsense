import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import seaborn as sns


def generate_gaussian_nb_plots(correlation=0.0):
    # Simulate correlated Gaussian data manually
    np.random.seed(42)
    mean_class0 = [0, 0]
    mean_class1 = [2, 2]
    cov = [[1, correlation], [correlation, 1]]

    X0 = np.random.multivariate_normal(mean_class0, cov, size=100)
    X1 = np.random.multivariate_normal(mean_class1, cov, size=100)

    X = np.vstack((X0, X1))
    y = np.array([0] * 100 + [1] * 100)

    # Fit model
    model = GaussianNB()
    model.fit(X, y)

    # Mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # --- Plot 1: Decision Surface ---
    fig1, ax1 = plt.subplots()
    ax1.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors='k')
    ax1.set_title("Decision Boundary (Naive Bayes)")

    # --- Plot 2: Feature-wise Gaussian PDFs ---
    fig2, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i, ax in enumerate(axs):
        sns.kdeplot(X0[:, i], label='Class 0', ax=ax, fill=True)
        sns.kdeplot(X1[:, i], label='Class 1', ax=ax, fill=True)
        ax.set_title(f"Feature {i + 1} Distribution")
        ax.legend()

    # --- Posterior Probabilities for Random Test Points ---
    test_points = np.array([[1, 1], [0, 2], [3, 1]])
    probs = model.predict_proba(test_points)

    return fig1, fig2, test_points, probs
