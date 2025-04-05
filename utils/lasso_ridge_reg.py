import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso

def plot_lasso_ridge(reg_type='ridge', alpha=1.0, manual=False, w=1.0, b=0.0, test_x=0.0):
    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X[:, 0] + np.random.randn(100)

    X_reshaped = X.reshape(-1, 1)

    if manual:
        y_pred = w * X[:, 0] + b
        test_y = w * test_x + b
    else:
        if reg_type == 'ridge':
            model = Ridge(alpha=alpha)
        else:
            model = Lasso(alpha=alpha, max_iter=10000)

        model.fit(X_reshaped, y)
        w = model.coef_[0]
        b = model.intercept_
        y_pred = model.predict(X_reshaped)
        test_y = model.predict([[test_x]])[0]

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', alpha=0.5, label='Data')
    ax.plot(X, y_pred, color='green', linewidth=2, label='Prediction Line')
    ax.plot(test_x, test_y, 'yo', markersize=10, label=f"Predicted: {test_y:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title(f"{reg_type.title()} Regression")
    ax.legend()

    return fig




