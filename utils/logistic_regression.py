import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

def apply_activation(z, activation='sigmoid'):
    if activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    elif activation == 'relu':
        return np.maximum(0, z)

def generate_logistic_regression_plot(activation='sigmoid'):
    # Generate 2D classification dataset
    X, y = make_classification(
        n_samples=100, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, random_state=42
    )

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X, y)

    # Mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # Predict probabilities
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], model.coef_.T) + model.intercept_
    Z = apply_activation(Z, activation=activation)
    Z = Z.reshape(xx.shape)

    # Plotting
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.7, levels=20, cmap="RdBu", vmin=0, vmax=1)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap="RdBu", s=50)
    ax.set_title(f"Logistic Regression Decision Boundary ({activation.capitalize()} Activation)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return fig
