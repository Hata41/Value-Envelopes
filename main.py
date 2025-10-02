
from __future__ import annotations

import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from typing import Optional, Tuple

from environments import TabularMDP
from utils import generate_dataset
from offline_rl import compute_offline_bounds
from online_rl import (
    StandardUCBVI,
    VShapingAgent,
    QShapingAgent,
    BonusShapingOnlyAgent,
    UpperBonusShapingAgent,
    UpperQShapingAgent,
    UpperVShapingAgent
)

def run_single_trial(params: tuple):
    # This function is unchanged
    algo_name, seed, mdp, T, delta, offline_bounds, K = params
    
    agent = None
    if 'Standard' in algo_name:
        agent = StandardUCBVI(mdp=mdp, T=T, delta=delta)
    elif 'Upper-V Shaping' in algo_name:
        agent = UpperVShapingAgent(mdp=mdp, offline_bounds=offline_bounds, T=T, delta=delta)
    elif 'Upper-Q Shaping' in algo_name:
        agent = UpperQShapingAgent(mdp=mdp, offline_bounds=offline_bounds, T=T, delta=delta)
    elif 'V-Shaping' in algo_name:
        agent = VShapingAgent(mdp=mdp, offline_bounds=offline_bounds, T=T, delta=delta)
    elif 'Q-Shaping' in algo_name:
        agent = QShapingAgent(mdp=mdp, offline_bounds=offline_bounds, T=T, delta=delta)
    elif 'Bonus-Shaping Only' in algo_name:
        agent = BonusShapingOnlyAgent(mdp=mdp, offline_bounds=offline_bounds, T=T, delta=delta)
    elif 'Upper-Bonus Shaping' in algo_name:
        agent = UpperBonusShapingAgent(mdp=mdp, offline_bounds=offline_bounds, T=T, delta=delta)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    try:
        np.random.seed(seed)
        regrets, _ = agent.run()
        full_name = algo_name if K is None else f"{algo_name} (K={K})"
        return full_name, np.cumsum(regrets)
    except Exception as e:
        print(f"Error in trial {algo_name} seed {seed}: {e}")
        return algo_name, np.full(T, np.nan)

def run_comparison_experiment(
    H: int = 4,
    S: int = 12,
    A: int = 3,
    T: int = int(1e3),
    delta: float = 0.05,
    intermediate_reward_range: Optional[Tuple[float, float]] = None,
    terminal_reward_range: Optional[Tuple[float, float]] = None,
    reward_sparse: bool = False,
    mdp_layered: bool = True,
    use_h_split: bool = True,
    K_values: list[int] = [120, 1200, 6000],
    n_seeds: int = 10,
    n_jobs: int = -1,
):
    if mdp_layered and S % H != 0: raise ValueError(f"Layered MDP needs S({S}) divisible by H({H}).")

    start_time = time.time()
    mdp_type_str = "Layered" if mdp_layered else "Standard"
    
    folder_name = f"RegretCurves_H{H}_S{S}_A{A}_T{T}_{mdp_type_str}"
    os.makedirs(folder_name, exist_ok=True)
    
    for k in K_values:
        if k > 0:
            os.makedirs(os.path.join(folder_name, f"K={k}"), exist_ok=True)

    reward_desc = []
    if intermediate_reward_range is not None:
        if intermediate_reward_range[0] == intermediate_reward_range[1] == 0.0:
            reward_desc.append("Intermediate=0")
        else:
            reward_desc.append(f"Intermediate=[{intermediate_reward_range[0]:.2f}, {intermediate_reward_range[1]:.2f}]")
    elif reward_sparse:
         reward_desc.append("Intermediate=0")
    else:
        reward_desc.append("Intermediate=Dense")
    if terminal_reward_range is not None:
        reward_desc.append(f"Terminal=[{terminal_reward_range[0]:.2f}, {terminal_reward_range[1]:.2f}]")
    elif reward_sparse:
        reward_desc.append("Terminal=Goal")
    reward_str = ", ".join(reward_desc)

    params_filepath = os.path.join(folder_name, "parameters.txt")
    with open(params_filepath, 'w') as f:
        f.write(f" MDP_Type: {mdp_type_str}\n"
                f" Reward_Structure: {reward_str}\n"
                f" Offline_Data_Split: {'H-Split' if use_h_split else 'Full_Dataset'}\n"
                f" H: {H}\n S: {S}\n A: {A}\n T: {T}\n"
                f" delta: {delta}\n K_values: {K_values}\n"
                f" n_seeds: {n_seeds}\n")
    
    master_seed = 42
    mdp = TabularMDP.random(
        H=H, S=S, A=A,
        layered=mdp_layered,
        seed=master_seed,
        reward_sparse=reward_sparse,
        intermediate_reward_range=intermediate_reward_range,
        terminal_reward_range=terminal_reward_range
    )
    
    all_tasks = []
    rng = np.random.default_rng(master_seed + 1)
    agent_seeds = rng.integers(0, 2**31, n_seeds)
    all_shaping_agents = ['V-Shaping', 
                          'Q-Shaping', 
                        #   'Bonus-Shaping Only'
                        ]
    
    for seed in agent_seeds:
        all_tasks.append(('Standard UCBVI', seed, mdp, T, delta, None, None))
        
    for K in K_values:
        if K <= 0 or K % H != 0: continue
        dataset = generate_dataset(mdp, K=K, seed=master_seed + K)
        offline_bounds = compute_offline_bounds(mdp, dataset, delta, seed=master_seed + K + 1, use_h_split=use_h_split)
        for algo in all_shaping_agents:
            for seed in agent_seeds:
                all_tasks.append((algo, seed, mdp, T, delta, offline_bounds, K))

    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_single_trial)(task) for task in all_tasks)

    processed_results = {}
    for name, trace in results:
        if name not in processed_results: processed_results[name] = []
        if not np.any(np.isnan(trace)): processed_results[name].append(trace)
    
    averaged_regret = {}
    for name, traces in processed_results.items():
        if not traces: continue
        stacked_traces = np.stack(traces) if len(traces) > 1 else np.array(traces)
        averaged_regret[name] = np.mean(stacked_traces, axis=0)

    fig, ax = plt.subplots(figsize=(14, 10))
    plot_filename = os.path.join(folder_name, f"regret_summary_{mdp_type_str}.png".lower())
    plt.savefig(plot_filename)

    num_points_to_save = 20
    indices = np.linspace(0, T - 1, num_points_to_save, dtype=int)
    for name, mean_regret in averaged_regret.items():
        subsampled_episodes = indices + 1
        subsampled_regret = mean_regret[indices]
        
        if "Standard UCBVI" in name:
            clean_name = "Standard_UCBVI"
            filepath = os.path.join(folder_name, f"{clean_name}.dat")
        else:
            match = re.match(r"(.+?)\s*\(K=(\d+)\)", name)
            if not match:
                print(f"  > Warning: Could not parse agent name '{name}'. Skipping file save.")
                continue
            
            algo_part, k_part = match.groups()
            clean_name = algo_part.replace(' ', '_').replace('-', '_')
            k_folder = f"K={k_part}"
            filepath = os.path.join(folder_name, k_folder, f"{clean_name}.dat")

        with open(filepath, 'w') as f:
            f.write("Episode CumulativeRegret\n")
            for ep, reg in zip(subsampled_episodes, subsampled_regret):
                f.write(f"{ep} {reg:.6f}\n")
        print(f"  > Saved {filepath}")

    print(f"\nExperiment complete. Total duration: {(time.time() - start_time):.2f} seconds.")


if __name__ == "__main__":
    H_val = 4
    K_base_values = [0.5, 1, 2, 3]
    
    run_comparison_experiment(
        H=H_val, S=12, A=3, T=int(1e6),
        mdp_layered=True,
        use_h_split=True,
        intermediate_reward_range=(0.0, 0.0),
        terminal_reward_range=(0.0, 1.0),        
        K_values=[int(H_val * k * 1e4) for k in K_base_values],
        n_seeds=1, # 4
        n_jobs=-1,
    )