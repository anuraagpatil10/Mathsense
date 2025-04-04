import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def generate_gaussian_nb_plot():
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=200, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, random_state=42
    )

    # Fit GaussianNB
    model = GaussianNB()
    model.fit(X, y)
    accuracy = accuracy_score(y, model.predict(X))

    # Create mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors='k')
    ax.set_title(f"Gaussian Naive Bayes Classification (Accuracy: {accuracy:.2f})")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return fig
