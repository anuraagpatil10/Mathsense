import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def generate_pca_comparison_plot(n_components=2):
    np.random.seed(42)
    mean = [0, 0, 0]
    cov = [[3, 1, 1], [1, 2, 0.5], [1, 0.5, 1.5]]
    X = np.random.multivariate_normal(mean, cov, size=200)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_.sum()

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c='teal', alpha=0.6)
    ax1.set_title("Original 3D Data")
    ax1.set_xlabel("X1")
    ax1.set_ylabel("X2")
    ax1.set_zlabel("X3")

    ax2 = fig.add_subplot(1, 2, 2)
    if n_components == 1:
        ax2.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), c='orange', alpha=0.7)
        ax2.set_xlabel("Principal Component 1")
        ax2.set_ylabel("Zero Axis")
        ax2.set_title(f"PCA to 1D (Variance: {explained_variance:.2%})")
    elif n_components == 2:
        ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='orange', alpha=0.7)
        ax2.set_xlabel("Principal Component 1")
        ax2.set_ylabel("Principal Component 2")
        ax2.set_title(f"PCA to 2D (Variance: {explained_variance:.2%})")
    else:
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c='orange', alpha=0.7)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_zlabel("PC3")
        ax2.set_title(f"PCA to 3D (Variance: {explained_variance:.2%})")

    plt.tight_layout()
    return fig

