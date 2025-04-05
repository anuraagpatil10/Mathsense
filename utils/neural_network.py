import matplotlib.pyplot as plt
import numpy as np

def draw_neural_net(ax, layer_sizes, weights=None, biases=None, activations=None):
    v_spacing = 1
    h_spacing = 2
    radius = 0.1
    ax.axis('off')

    # Store neuron coordinates for edge drawing
    neuron_coords = []

    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2
        coords = []

        for j in range(layer_size):
            x = i * h_spacing
            y = layer_top - j * v_spacing
            coords.append((x, y))

            # Color by layer
            if i == 0:
                color = 'orchid'        # input
            elif i == len(layer_sizes) - 1:
                color = 'turquoise'     # output
            else:
                color = 'skyblue'       # hidden

            # Neuron circle
            circle = plt.Circle((x, y), radius, color=color, zorder=4)
            ax.add_artist(circle)

            # Neuron label
            label = ''
            if i == 0:
                label = f"$x_{{{j+1}}}$"
            elif i == len(layer_sizes) - 1:
                label = f"$y_{{{j+1}}}$"
            else:
                label = f"$h_{{{i},{j+1}}}$"

            ax.text(x, y, label, fontsize=8, ha='center', va='center', zorder=5, color='white')

            # Activation value
            try:
                val = float(activations[i][j])
                ax.text(x, y - 0.2, f"{val:.2f}", fontsize=7, ha='center', va='center', color='black')
            except Exception as e:
                ax.text(x, y - 0.2, "-", fontsize=7, ha='center', va='center', color='black')
            # if activations and len(activations) > i:
            #     val = f"{activations[i][j]:.2f}"
            #     ax.text(x, y - 0.2, val, fontsize=7, ha='center', va='center', color='black')

        neuron_coords.append(coords)

    # Draw edges with weight labels
    for i in range(len(layer_sizes) - 1):
        for j, (x1, y1) in enumerate(neuron_coords[i]):
            for k, (x2, y2) in enumerate(neuron_coords[i+1]):
                ax.plot([x1, x2], [y1, y2], 'gray', lw=0.5, alpha=0.5)

                if weights:
                    weight_val = weights[i][j, k]
                    mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
                    ax.text(mid_x, mid_y, f"{weight_val:.1f}", fontsize=6, color='red', alpha=0.7)

        # Biases (drawn next to receiving neuron)
        if biases:
            for k, (x, y) in enumerate(neuron_coords[i+1]):
                ax.text(x + 0.2, y + 0.1, f"b={biases[i][k]:.1f}", fontsize=6, color='green')
