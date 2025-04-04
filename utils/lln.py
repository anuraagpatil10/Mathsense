import numpy as np
import matplotlib.pyplot as plt

def generate_population(distribution, size=10000):
    if distribution == "Normal":
        return np.random.normal(loc=0, scale=1, size=size)
    elif distribution == "Uniform":
        return np.random.uniform(low=-3, high=3, size=size)
    elif distribution == "Exponential":
        return np.random.exponential(scale=1.0, size=size)

def generate_lln_plot(distribution, trials):
    data = generate_population(distribution, size=trials)
    cumulative_avg = np.cumsum(data) / np.arange(1, trials + 1)

    fig, ax = plt.subplots()
    ax.plot(cumulative_avg, label="Cumulative Average")
    ax.axhline(np.mean(data), color='red', linestyle='--', label="Expected Value")
    ax.set_title(f"Law of Large Numbers ({distribution})")
    ax.set_xlabel("Number of Trials")
    ax.set_ylabel("Running Mean")
    ax.legend()
    return fig
