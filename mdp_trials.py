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
        'V-Shaping', 'Q-Shaping', 'Bonus-Shaping Only',
        'Upper-Bonus Shaping', 'Upper-V Shaping', 'Upper-Q Shaping'
    ]
    final_performance: Dict[str, list] = {name: [] for name in shaping_algos}

    for x in x_values:
        terminal_range = reward_lambda(x)

        if terminal_range[1] <= terminal_range[0]:
            print(f"\n[Skipping invalid range for x = {x:.4f}]")
            for name in shaping_algos:
                final_performance[name].append(np.nan)
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

        for seed in agent_seeds:
            all_tasks.append(('Standard UCBVI', seed, mdp, T, 0.05, None, None))
        for algo_name in shaping_algos:
            for seed in agent_seeds:
                all_tasks.append((algo_name, seed, mdp, T, 0.05, offline_bounds, K))

        results = Parallel(n_jobs=n_jobs, verbose=0)(delayed(run_single_trial)(task) for task in all_tasks)

        processed_regrets: Dict[str, list] = {}
        for name, trace in results:
            base_name = name.split(' (K=')[0]
            if base_name not in processed_regrets: processed_regrets[base_name] = []
            if not np.any(np.isnan(trace)):
                processed_regrets[base_name].append(trace[-1])
        
        mean_final_regret: Dict[str, float] = {}
        for name, regret_values in processed_regrets.items():
            if regret_values:
                mean_final_regret[name] = np.mean(regret_values)

        baseline_regret = mean_final_regret.get('Standard UCBVI')
        if baseline_regret is None or baseline_regret < 1e-6:
            for name in shaping_algos:
                final_performance[name].append(np.nan)
            continue
            
        for name in shaping_algos:
            algo_regret = mean_final_regret.get(name)
            if algo_regret is not None:
                improvement = (baseline_regret - algo_regret) / baseline_regret
                final_performance[name].append(improvement)
            else:
                final_performance[name].append(np.nan)

    for name, perfs in final_performance.items():
        clean_name = name.replace(' ', '_').replace('-', '_')
        filename = f"{clean_name}.dat"
        filepath = os.path.join(folder_name, filename)
        
        with open(filepath, 'w') as f:
            f.write("# X_Value Improvement_Ratio\n")
            for i, perf_val in enumerate(perfs):
                if not np.isnan(perf_val):
                    x_val = plot_x_variable[i]
                    f.write(f"{x_val:.8f} {perf_val:.8f}\n")
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

def setup_shrinking_reward_experiment(params):
    runner_params = params.copy()
    num_points = runner_params.pop('num_points')

    gaps = np.geomspace(1.0, 0.01, num_points)
    x_values = np.sort(1.0 - gaps)
    run_and_save_experiment(
        experiment_name="ShrinkingReward",
        x_values=x_values,
        reward_lambda=lambda x: (0.0, 1.0 - x),
        plot_x_variable=1.0 - x_values,
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
        'n_seeds': 10,
        'n_jobs': -1,
        'num_points': 30
    }

    setup_expanding_reward_experiment(common_params)
    setup_shrinking_reward_experiment(common_params)
    setup_sliding_reward_experiment(common_params, reward_width=0.1)
    