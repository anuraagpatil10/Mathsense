import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def plot_logistic_regression(C=1.0, manual=False, w0=1.0, w1=1.0, bias=0.0, test_point=(0, 0)):
    # Generate data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1, random_state=42)

    if manual:
        def custom_predict(X):
            z = w0 * X[:, 0] + w1 * X[:, 1] + bias
            return sigmoid(z)

        proba = custom_predict(X)
        preds = (proba >= 0.5).astype(int)
    else:
        model = LogisticRegression(C=C, solver='lbfgs')
        model.fit(X, y)
        w0, w1 = model.coef_[0]
        bias = model.intercept_[0]
        proba = model.predict_proba(X)[:, 1]
        preds = model.predict(X)

    # Plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = sigmoid(w0 * grid[:, 0] + w1 * grid[:, 1] + bias)
    zz = zz.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, zz, levels=20, cmap='coolwarm', alpha=0.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['blue', 'red']), edgecolor='k', alpha=0.8)
    ax.set_title("Logistic Regression Decision Surface")

    # Draw test point
    tx, ty = test_point
    z_test = sigmoid(w0 * tx + w1 * ty + bias)
    ax.plot(tx, ty, marker='o', markersize=10, markerfacecolor='yellow', markeredgecolor='black')
    ax.text(tx + 0.2, ty, f"P(y=1): {z_test:.2f}", fontsize=10, color='black')

    return fig




