import numpy as np

def compute_binary_maxcut(graph, weights, z):
    """Compute binary MaxCut objective for a given partition."""
    cut_value = 0
    for (i, j), w in zip(graph, weights):
        if z[i-1] != z[j-1]:  # Edge crosses partition
            cut_value += w
    return cut_value

# Graph setup (reusing prior generation logic)
def generate_g1_inspired_graph(n=20, edge_prob=0.4, seed=42):
    np.random.seed(seed)
    graph = []
    weights = []
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            if np.random.rand() < edge_prob:
                graph.append((i, j))
                weights.append(np.random.choice([1, -1], p=[0.5, 0.5]))
    return graph, weights

# Final state and binarization
final_state = np.array([0.0542, 0.0501, 0.0451, 0.0544, 0.0461, 0.0482, 0.0511, 0.0473, 0.0524, 0.0468,
                        0.0531, 0.0492, 0.0459, 0.0540, 0.0465, 0.0489, 0.0507, 0.0476, 0.0520, 0.0497])
z = (final_state >= np.median(final_state)).astype(int)
graph, weights = generate_g1_inspired_graph()
binary_cut = compute_binary_maxcut(graph, weights, z)
print(f"Binary MaxCut Objective: {binary_cut}")