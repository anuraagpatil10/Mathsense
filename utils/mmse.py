import numpy as np
import matplotlib.pyplot as plt

def generate_mmse_plot(noise_level):
    np.random.seed(42)
    true_signal = np.linspace(0, 10, 500)
    noisy_signal = true_signal + np.random.normal(0, noise_level, true_signal.shape)

    # MMSE estimation via moving average
    window_size = 20
    mmse_signal = np.convolve(noisy_signal, np.ones(window_size)/window_size, mode='same')

    fig, ax = plt.subplots()
    ax.plot(true_signal, label="True Signal", linewidth=1)
    ax.plot(noisy_signal, label="Noisy Observation", alpha=0.4)
    ax.plot(mmse_signal, label="MMSE Estimate", linewidth=2)
    ax.set_title("MMSE Estimation on Noisy Signal")
    ax.legend()
    return fig
