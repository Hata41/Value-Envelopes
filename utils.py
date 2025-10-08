from __future__ import annotations

import numpy as np
from typing import Callable, Iterable, List, Tuple, Optional
from joblib import Parallel, delayed
import numba

from environments import TabularMDP

def value_iteration(mdp: TabularMDP) -> Tuple[np.ndarray, np.ndarray]:
    H, S, A = mdp.H, mdp.S, mdp.A
    V = np.zeros((H + 1, S), dtype=float)
    pi = np.zeros((H, S), dtype=int)
    for h in reversed(range(H)):
        for s in range(S):
            q_vals = np.zeros(A, dtype=float)
            for a in range(A):
                # One‐step return plus value of next state
                q_vals[a] = mdp.R[h, s, a] + float(np.dot(mdp.P[h, s, a], V[h + 1]))
            V[h, s] = float(np.max(q_vals))
            pi[h, s] = int(np.argmax(q_vals))
    return V, pi


def simulate_episode(mdp: TabularMDP,
                     policy: Callable[[int, int], int]) -> Tuple[List[int], List[int], List[float]]:
    states: List[int] = []
    actions: List[int] = []
    rewards: List[float] = []
    s = mdp.reset()
    states.append(int(s))
    for h in range(mdp.H):
        a = int(policy(h, s))
        next_s, r = mdp.step(a)
        actions.append(a)
        rewards.append(float(r))
        s = next_s
        states.append(int(s))
    return states, actions, rewards


def evaluate_policy(mdp: TabularMDP, policy: np.ndarray) -> np.ndarray:
    H, S = mdp.H, mdp.S
    V_pi = np.zeros((H + 1, S), dtype=float)
    for h in reversed(range(H)):
        for s in range(S):
            a = int(policy[h, s])
            V_pi[h, s] = mdp.R[h, s, a] + float(
                np.dot(mdp.P[h, s, a], V_pi[h + 1])
            )
    return V_pi


def _generate_single_trajectory(
    mdp_params: Tuple,
    behaviour_policy: Optional[Callable[[int, int], int]] = None,
    seed: Optional[int] = None
) -> Tuple[List[int], List[int], List[float]]:

    H, S, A, P, R, rho = mdp_params
    rng = np.random.default_rng(seed)

    local_mdp = TabularMDP(H=H, S=S, A=A, P=P, R=R, rho=rho)

    def default_policy(h: int, s: int) -> int:
        return int(rng.integers(low=0, high=A))
    
    policy_to_use = behaviour_policy or default_policy

    s = local_mdp.reset()
    states = [int(s)]
    actions = []
    rewards = []
    for h in range(H):
        a = int(policy_to_use(h, s))
        next_s, r = local_mdp.step(a)
        actions.append(a)
        rewards.append(float(r))
        s = next_s
        states.append(int(s))
    return states, actions, rewards


def generate_dataset(mdp: TabularMDP, K: int,
                     behaviour_policy: Optional[Callable[[int, int], int]] = None,
                     seed: Optional[int] = None,
                     n_jobs: int = -1) -> List[Tuple[List[int], List[int], List[float]]]:

    assert K % mdp.H == 0, (
        f"Number of trajectories K={K} must be a multiple of the horizon H={mdp.H}")

    # Package the MDP parameters to be passed to parallel workers
    mdp_params = (mdp.H, mdp.S, mdp.A, mdp.P, mdp.R, mdp.rho)
    
    # Generate unique seeds for each trajectory for reproducibility
    rng = np.random.default_rng(seed)
    seeds = rng.integers(low=0, high=2**31, size=K)

    print(f"Generating {K} trajectories using {n_jobs if n_jobs > 0 else 'all'} CPU cores...")
    
    # Use joblib to parallelize the generation of trajectories
    trajectories = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_generate_single_trajectory)(mdp_params, behaviour_policy, seed=s)
        for s in seeds
    )
    
    return trajectories

@numba.jit(nopython=True, cache=True, parallel=True)
def _jit_value_iteration_core(H, S, A, R, P):
    V = np.zeros((H + 1, S), dtype=np.float64)
    pi = np.zeros((H, S), dtype=np.int64)
    
    for h in range(H - 1, -1, -1):
        for s in numba.prange(S): # Parallelize the state loop
            q_vals = np.zeros(A, dtype=np.float64)
            for a in range(A):
                expected_v = 0.0
                for s_next in range(S):
                    expected_v += P[h, s, a, s_next] * V[h + 1, s_next]
                q_vals[a] = R[h, s, a] + expected_v
            
            best_a = 0
            max_q = -np.inf
            for a in range(A):
                if q_vals[a] > max_q:
                    max_q = q_vals[a]
                    best_a = a
            
            V[h, s] = max_q
            pi[h, s] = best_a
    return V, pi

def value_iteration(mdp: TabularMDP) -> Tuple[np.ndarray, np.ndarray]:
    return _jit_value_iteration_core(mdp.H, mdp.S, mdp.A, mdp.R, mdp.P)


@numba.jit(nopython=True, cache=True, parallel=True)
def _jit_evaluate_policy_core(H, S, A, R, P, policy):
    V_pi = np.zeros((H + 1, S), dtype=np.float64)
    
    for h in range(H - 1, -1, -1):
        for s in numba.prange(S): # Parallelize the state loop
            a = policy[h, s]
            expected_v = 0.0
            for s_next in range(S):
                expected_v += P[h, s, a, s_next] * V_pi[h + 1, s_next]
            V_pi[h, s] = R[h, s, a] + expected_v
    return V_pi

def evaluate_policy(mdp: TabularMDP, policy: np.ndarray) -> np.ndarray:
    return _jit_evaluate_policy_core(mdp.H, mdp.S, mdp.A, mdp.R, mdp.P, policy)

def calculate_behavioral_visitation(mdp: TabularMDP) -> float:
    H, S, A = mdp.H, mdp.S, mdp.A
    P = mdp.P
    
    state_dist = np.zeros((H, S), dtype=np.float64)
    state_dist[0, :] = mdp.rho
    
    for h in range(H - 1):
        next_state_dist = np.zeros(S, dtype=np.float64)
        for s in range(S):
            if state_dist[h, s] > 0:
                for a in range(A):
                    # P(S_h=s) * pi_b(a|s) * P(S_{h+1}|s,a)
                    next_state_dist += state_dist[h, s] * (1.0 / A) * P[h, s, a, :]
        state_dist[h + 1, :] = next_state_dist

    d_b_min = np.inf
    for h in range(H):
        for s in range(S):
            visitation_prob = state_dist[h, s] / A
            if 1e-12 < visitation_prob < d_b_min:
                d_b_min = visitation_prob
    
    return d_b_min if d_b_min != np.inf else 1e-12