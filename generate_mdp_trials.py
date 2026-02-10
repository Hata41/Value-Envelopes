from __future__ import annotations

import os
import numpy as np
from joblib import Parallel, delayed
from typing import Tuple, Dict, Callable

from environments import TabularMDP
from utils import generate_dataset
from offline_rl import compute_offline_bounds
from online_rl import (
    StandardUCBVI,
    CountInitializedUCBVI,
    VShapingAgent,
    QShapingAgent,
    BonusShapingOnlyAgent,
    UpperBonusShapingAgent,
    UpperQShapingAgent,
    UpperVShapingAgent
)

def run_single_trial(params: tuple):
    algo_name, seed, mdp, T, delta, offline_bounds, K = params
    
    agent = None
    if 'Standard' in algo_name:
        agent = StandardUCBVI(mdp=mdp, T=T, delta=delta)
    elif 'Count-Init UCBVI' in algo_name:
        agent = CountInitializedUCBVI(mdp=mdp, offline_bounds=offline_bounds, T=T, delta=delta)
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
        # Return the last cumulative regret value
        return full_name, np.cumsum(regrets)[-1]
    except Exception as e:
        print(f"Error in trial {algo_name} seed {seed}: {e}")
        return algo_name, np.nan

def run_and_save_experiment(
    experiment_name: str,
    x_values: np.ndarray,
    reward_lambda: Callable[[float], Tuple[float, float]],
    plot_x_variable: np.ndarray,
    H: int, S: int, A: int, T: int, K: int,
    n_seeds: int, n_jobs: int
):
    folder_name = experiment_name
    os.makedirs(folder_name, exist_ok=True)
    
    params_filepath = os.path.join(folder_name, "parameters.txt")
    params_content = (
        f"Experiment_Name: {experiment_name}\n"
        f"Horizon_H: {H}\n"
        f"States_S: {S}\n"
        f"Actions_A: {A}\n"
        f"Online_Episodes_T: {T}\n"
        f"Offline_Trajectories_K: {K}\n"
        f"Number_of_Seeds: {n_seeds}\n"
        f"Number_of_Jobs: {n_jobs}\n"
    )
    with open(params_filepath, 'w') as f:
        f.write(params_content)

    print(f"Results will be saved in: '{folder_name}/'")
    print(f"Parameters for this run saved in: '{params_filepath}'")

    shaping_algos = [
        'Bonus-Shaping Only',
        'Upper-Bonus Shaping', 
        'Count-Init UCBVI'
    ]
    
    # Structure: algo_name -> list of tuples (mean, std) corresponding to x_values
    final_performance: Dict[str, list] = {name: [] for name in shaping_algos}

    for x in x_values:
        terminal_range = reward_lambda(x)

        if terminal_range[1] <= terminal_range[0]:
            print(f"\n[Skipping invalid range for x = {x:.4f}]")
            for name in shaping_algos:
                final_performance[name].append((np.nan, np.nan))
            continue

        print(f"\n[Running for x={x:.4f}, R_term=({terminal_range[0]:.3f}, {terminal_range[1]:.3f})]")
        
        mdp_seed = int(x * 100000)
        mdp = TabularMDP.random(H=H, S=S, A=A, layered=True, seed=mdp_seed,
                                intermediate_reward_range=(0.0, 0.0),
                                terminal_reward_range=terminal_range)
        
        all_tasks = []
        rng = np.random.default_rng(mdp_seed + 1)
        agent_seeds = rng.integers(0, 2**31, n_seeds)
        dataset = generate_dataset(mdp, K=K, seed=mdp_seed + 2)
        offline_bounds = compute_offline_bounds(mdp, dataset, delta=0.05, seed=mdp_seed + 3)

        # 1. Schedule Standard UCBVI (Baselines)
        for seed in agent_seeds:
            all_tasks.append(('Standard UCBVI', seed, mdp, T, 0.05, None, None))
            
        # 2. Schedule Shaping Algorithms
        for algo_name in shaping_algos:
            for seed in agent_seeds:
                all_tasks.append((algo_name, seed, mdp, T, 0.05, offline_bounds, K))

        results = Parallel(n_jobs=n_jobs, verbose=0)(delayed(run_single_trial)(task) for task in all_tasks)

        # Organize results by Algorithm -> List of Regrets (ordered by seed)
        # Note: Parallel preserves the order of input `all_tasks` in `results`.
        # Since we appended seeds in the same order for every algo, index i corresponds to seed i.
        
        regrets_by_algo: Dict[str, list] = {}
        
        for name, final_regret in results:
            base_name = name.split(' (K=')[0]
            if base_name not in regrets_by_algo:
                regrets_by_algo[base_name] = []
            regrets_by_algo[base_name].append(final_regret)

        # Extract baseline regrets
        baseline_regrets = np.array(regrets_by_algo.get('Standard UCBVI', []))
        
        # Calculate stats for each shaping algo
        for name in shaping_algos:
            algo_regrets = np.array(regrets_by_algo.get(name, []))
            
            if len(baseline_regrets) != n_seeds or len(algo_regrets) != n_seeds:
                # If missing data (e.g. errors), append NaNs
                final_performance[name].append((np.nan, np.nan))
                continue

            # Filter out any NaN runs (if individual trials failed)
            valid_mask = (~np.isnan(baseline_regrets)) & (~np.isnan(algo_regrets))
            
            if not np.any(valid_mask):
                final_performance[name].append((np.nan, np.nan))
                continue
                
            valid_base = baseline_regrets[valid_mask]
            valid_algo = algo_regrets[valid_mask]
            
            # Prevent division by zero
            valid_base[valid_base < 1e-6] = 1e-6
            
            # Calculate improvement per seed
            improvements = (valid_base - valid_algo) / valid_base
            
            mean_imp = np.mean(improvements)
            std_imp = np.std(improvements)
            
            final_performance[name].append((mean_imp, std_imp))

    # Write to files
    for name, stats in final_performance.items():
        clean_name = name.replace(' ', '_').replace('-', '_')
        filename = f"{clean_name}.dat"
        filepath = os.path.join(folder_name, filename)
        
        with open(filepath, 'w') as f:
            f.write("# X_Value Mean_Improvement Std_Improvement\n")
            for i, (mean_val, std_val) in enumerate(stats):
                if not np.isnan(mean_val):
                    x_val = plot_x_variable[i]
                    f.write(f"{x_val:.8f} {mean_val:.8f} {std_val:.8f}\n")
        print(f"  > Saved {filepath}")

def setup_expanding_reward_experiment(params):
    runner_params = params.copy()
    num_points = runner_params.pop('num_points')

    x_values = np.geomspace(0.01, 1.0, num_points)
    run_and_save_experiment(
        experiment_name="ExpandingReward",
        x_values=x_values,
        reward_lambda=lambda x: (1.0 - x, 1.0),
        plot_x_variable=x_values,
        **runner_params
    )

def setup_sliding_reward_experiment(params, reward_width):
    runner_params = params.copy()
    num_points = runner_params.pop('num_points')

    x_values = np.linspace(0.0, 1.0 - reward_width, num_points)
    run_and_save_experiment(
        experiment_name=f"SlidingWindow_width{reward_width:.2f}".replace('.', 'p'),
        x_values=x_values,
        reward_lambda=lambda x: (x, x + reward_width),
        plot_x_variable=x_values,
        **runner_params
    )


if __name__ == "__main__":
    common_params = {
        'H': 3,
        'S': 9,
        'A': 3,
        'T': int(1e6),
        'K': int(6e4),
        'n_seeds': 4, # Increased seeds to make Std Dev meaningful
        'n_jobs': -1,
        'num_points': 10 # Low for testing, increase for real plot
    }

    setup_expanding_reward_experiment(common_params)
    setup_sliding_reward_experiment(common_params, reward_width=0.1)