import numpy as np
import matplotlib.pyplot as plt

def generate_population(distribution, size=10000):
    if distribution == "Normal":
        return np.random.normal(loc=0, scale=1, size=size)
    elif distribution == "Uniform":
        return np.random.uniform(low=-3, high=3, size=size)
    elif distribution == "Exponential":
        return np.random.exponential(scale=1.0, size=size)

def generate_clt_plot(distribution, sample_size, num_samples):
    population = generate_population(distribution)
    means = [np.mean(np.random.choice(population, size=sample_size)) for _ in range(num_samples)]

    fig, ax = plt.subplots()
    ax.hist(means, bins=40, color='skyblue', edgecolor='black')
    ax.set_title(f"Sampling Distribution of the Mean ({distribution})")
    ax.set_xlabel("Sample Mean")
    ax.set_ylabel("Frequency")
    return fig
