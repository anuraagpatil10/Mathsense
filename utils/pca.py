import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def generate_pca_plot(n_samples, n_features, n_components):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    X += np.random.normal(0, 0.5, X.shape)  # Add some noise

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots()
    if n_components == 2:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
        ax.set_title("PCA - First 2 Components")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
    else:
        ax.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        ax.set_title("Cumulative Explained Variance")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Explained Variance Ratio")
    return fig
