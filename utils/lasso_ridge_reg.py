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



# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Lasso, Ridge
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
#
# def generate_lasso_ridge_plot(regression_type="Lasso", alpha=1.0):
#     # Generate synthetic data
#     np.random.seed(42)
#     X = np.linspace(0, 10, 100).reshape(-1, 1)
#     y = 3 * X.squeeze() + np.random.randn(100) * 2  # Linear relationship with noise
#
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Choose model
#     if regression_type == "Lasso":
#         model = Lasso(alpha=alpha)
#     else:
#         model = Ridge(alpha=alpha)
#
#     # Train model
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     # Compute loss
#     mse = mean_squared_error(y_test, y_pred)
#
#     # Plot
#     fig, ax = plt.subplots()
#     ax.scatter(X_train, y_train, color="blue", label="Training Data", alpha=0.6)
#     ax.scatter(X_test, y_test, color="red", label="Test Data", alpha=0.6)
#     ax.plot(X_test, y_pred, color="black", label=f"{regression_type} Regression (α={alpha})")
#     ax.set_title(f"{regression_type} Regression (MSE: {mse:.2f})")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.legend()
#     return fig
