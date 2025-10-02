"""
utils.py
========

Utility routines for finite‑horizon tabular reinforcement learning.

This module provides functions for value iteration to compute the
ground‑truth optimal value function and policy for a given
``TabularMDP``, simulation of episodes under a specified policy,
and generation of offline datasets using a behaviour policy.  These
utilities are kept separate from the learning algorithms to keep
concerns modular.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Iterable, List, Tuple, Optional
from joblib import Parallel, delayed
import numba

from environments import TabularMDP

def value_iteration(mdp: TabularMDP) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the optimal value function and greedy policy via backward DP.

    Parameters
    ----------
    mdp: TabularMDP
        The environment for which to compute optimal values.

    Returns
    -------
    V_opt: np.ndarray of shape (H+1, S)
        Optimal state value for each step and state.  The last row
        ``V_opt[H]`` corresponds to the terminal value (all zeros).
    pi_opt: np.ndarray of shape (H, S)
        Optimal deterministic policy: ``pi_opt[h][s]`` is the action
        achieving the maximum value at step h in state s.

    Notes
    -----
    This implementation assumes rewards are already clipped to
    ``[0,1]``.  No discounting is used; the objective is the sum of
    rewards over the horizon.
    """
    H, S, A = mdp.H, mdp.S, mdp.A
    V = np.zeros((H + 1, S), dtype=float)
    pi = np.zeros((H, S), dtype=int)
    # Terminal value function V[H] is zero
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
    """Roll out a single episode following a (possibly stochastic) policy.

    Parameters
    ----------
    mdp: TabularMDP
        Environment to interact with.
    policy: Callable[[int, int], int]
        A function mapping (step, state) to an action.

    Returns
    -------
    states: list[int] of length H+1
        The sequence of visited states, including the terminal state.
    actions: list[int] of length H
        The sequence of actions chosen.
    rewards: list[float] of length H
        The rewards obtained at each step.
    """
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
    """Evaluate a deterministic policy on a tabular finite‑horizon MDP.

    Parameters
    ----------
    mdp: TabularMDP
        The environment; only the reward and transition matrices are
        used.
    policy: np.ndarray of shape (H, S)
        Deterministic policy mapping step and state to an action.

    Returns
    -------
    V_pi: np.ndarray of shape (H + 1, S)
        The value function of the policy: ``V_pi[h][s]`` is the
        expected return when starting from state ``s`` at step ``h``
        and following the given policy thereafter.  The final row
        ``V_pi[H]`` is identically zero.

    Notes
    -----
    This evaluation uses the true transition probabilities and
    reward function from ``mdp``; it should not be used for
    planning but only for measuring regret.
    """
    H, S = mdp.H, mdp.S
    V_pi = np.zeros((H + 1, S), dtype=float)
    # Terminal values are zero by initialisation
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
    """
    Helper function to generate one trajectory for parallel processing.
    It creates a local MDP instance to ensure thread safety.
    """
    H, S, A, P, R, rho = mdp_params
    rng = np.random.default_rng(seed)

    # Each parallel worker gets its own MDP instance
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
    """
    Generates an offline dataset using parallel trajectory rollouts.
    """
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
    """Numba-jitted core for optimal value iteration."""
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
    """Computes the optimal value function by calling the JIT core."""
    return _jit_value_iteration_core(mdp.H, mdp.S, mdp.A, mdp.R, mdp.P)


@numba.jit(nopython=True, cache=True, parallel=True)
def _jit_evaluate_policy_core(H, S, A, R, P, policy):
    """Numba-jitted core for policy evaluation."""
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
    """Evaluates a policy by calling the JIT core."""
    return _jit_evaluate_policy_core(mdp.H, mdp.S, mdp.A, mdp.R, mdp.P, policy)

def calculate_behavioral_visitation(mdp: TabularMDP) -> float:
    """
    Calculates the minimum non-zero state-action visitation probability
    d_b_min under a uniform random behavior policy.
    """
    H, S, A = mdp.H, mdp.S, mdp.A
    P = mdp.P
    
    # state_dist[h, s] will be P(S_h = s) under the behavior policy
    state_dist = np.zeros((H, S), dtype=np.float64)
    
    # The distribution for the first state (at step h=0) is the MDP's initial distribution
    state_dist[0, :] = mdp.rho
    
    # Forward pass to compute state distributions for all subsequent steps
    for h in range(H - 1):
        # Calculate the distribution for step h+1
        next_state_dist = np.zeros(S, dtype=np.float64)
        for s in range(S):
            if state_dist[h, s] > 0:
                for a in range(A):
                    # P(S_h=s) * pi_b(a|s) * P(S_{h+1}|s,a)
                    next_state_dist += state_dist[h, s] * (1.0 / A) * P[h, s, a, :]
        state_dist[h + 1, :] = next_state_dist

    # Now calculate d_h(s,a) = P(S_h=s) * pi_b(a|s) and find the min non-zero
    d_b_min = np.inf
    for h in range(H):
        for s in range(S):
            visitation_prob = state_dist[h, s] / A
            if 1e-12 < visitation_prob < d_b_min:
                d_b_min = visitation_prob
    
    return d_b_min if d_b_min != np.inf else 1e-12