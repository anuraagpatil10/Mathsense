import numpy as np
import matplotlib.pyplot as plt

def generate_bayes_plot(prior_A, prob_B_given_A, prob_B_given_not_A):
    posterior_values = []
    prob_B_range = np.linspace(0.01, 1, 100)

    for prob_B in prob_B_range:
        # Compute Bayes' Theorem: P(A|B) = [P(B|A) * P(A)] / P(B)
        prob_B_total = (prob_B_given_A * prior_A) + (prob_B_given_not_A * (1 - prior_A))
        posterior_A_given_B = (prob_B_given_A * prior_A) / prob_B_total
        posterior_values.append(posterior_A_given_B)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(prob_B_range, posterior_values, color="blue", label="Posterior P(A|B)")
    ax.axhline(y=prior_A, linestyle="dashed", color="red", label="Prior P(A)")
    ax.set_title("Bayes' Theorem: Posterior Probability vs Evidence")
    ax.set_xlabel("P(B) - Evidence Probability")
    ax.set_ylabel("P(A|B) - Posterior Probability")
    ax.legend()
    return fig
