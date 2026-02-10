
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from environments import TabularMDP
from utils import value_iteration, generate_dataset
from offline_rl import compute_offline_bounds


def export_to_pgfplots_dat(h_specific, K_values, results, span_h_layer, states_in_layer):
    filename = f"R_vs_D.dat"
    
    header_parts = ["K"]
    for s in states_in_layer:
        header_parts.append(f"D_s{s}")
    header_parts.extend(["R_h_layer", "span_h_layer"])
    header = " ".join(header_parts)

    with open(filename, 'w') as f:
        f.write(header + "\n")
        
        for i, K in enumerate(K_values):
            data_line = [str(K)]
            
            for s in states_in_layer:
                data_line.append(f"{results['D_values'][s][i]:.6f}")

            data_line.append(f"{results['R_h_layer'][i]:.6f}")

            data_line.append(f"{span_h_layer:.6f}")
            
            f.write(" ".join(data_line) + "\n")
            
    print(f"  > Exported data to {filename}")

def generate_width_data(h_specific: int):
    H = 4
    S = 12
    A = 3
    delta = 0.05
    mdp_seed = 42
    K_max = int(1e5)
    K_min = int(1e3)
    num_K_points = 25
    K_values = sorted(list(set([k - (k % H) for k in np.linspace(K_min, K_max, num=num_K_points, dtype=int) if k > 0])))
    K_max_actual = K_values[-1]
    S_layer = S // H

    if not (0 <= h_specific < H):
        raise ValueError(f"Invalid h_specific. Must be between 0 and H-1 ({H-1}).")

    mdp = TabularMDP.random(H=H, S=S, A=A, layered=True, seed=mdp_seed,
               )
    V_opt, _ = value_iteration(mdp)
    s_start_layer = h_specific * S_layer
    s_end_layer = s_start_layer + S_layer
    states_in_layer = range(s_start_layer, s_end_layer)
    v_opt_h_layer = V_opt[h_specific, s_start_layer:s_end_layer]
    span_h_layer = np.max(v_opt_h_layer) - np.min(v_opt_h_layer)

    large_dataset = generate_dataset(mdp, K=K_max_actual, seed=mdp_seed + 1, n_jobs=-1)

    results = {'R_h_layer': [], 'D_values': {s: [] for s in states_in_layer}}
    for K in tqdm(K_values, desc="Processing subsets"):
        subset_dataset = large_dataset[:K]
        offline_bounds = compute_offline_bounds(mdp, subset_dataset, delta,
                                                 use_h_split=True, seed=K + mdp_seed)
        
        u_vals = offline_bounds['U_values'][h_specific][s_start_layer:s_end_layer]
        w_vals = offline_bounds['W_values'][h_specific][s_start_layer:s_end_layer]
        results['R_h_layer'].append(np.max(u_vals) - np.min(w_vals))
        
        for s_idx, s in enumerate(states_in_layer):
            results['D_values'][s].append(offline_bounds['D'][h_specific][s])

    export_to_pgfplots_dat(h_specific, K_values, results, span_h_layer, states_in_layer)
    
    plt.style.use('default')


if __name__ == "__main__":
    h_to_analyze = 2
    generate_width_data(h_specific=h_to_analyze)