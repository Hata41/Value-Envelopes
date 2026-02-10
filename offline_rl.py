from __future__ import annotations

import numpy as np
import numba
from typing import List, Tuple, Dict, Any

from environments import TabularMDP


@numba.jit(nopython=True, cache=True, parallel=True)
def _jit_offline_bounds_core(H, S, A, R, P_hat, N_sa, delta):
    U_values = np.zeros((H + 1, S), dtype=np.float64)
    W_values = np.zeros((H + 1, S), dtype=np.float64)
    U_Q = np.zeros((H, S, A), dtype=np.float64)

    L1 = np.log((8.0 * S * A * H) / max(delta, 1e-12))
    c1, c2 = 2.0, 14.0 / 3.0

    for h in range(H - 1, -1, -1):
        q_lower_h = np.zeros((S, A), dtype=np.float64)
        for s in numba.prange(S):
            for a in range(A):
                n = N_sa[h, s, a]
                b = 0.0
                if n <= 1:
                    b = float(H - h)
                else:
                    mu_U, mu_W = 0.0, 0.0
                    for s_next in range(S):
                        p_s_next = P_hat[h, s, a, s_next]
                        mu_U += p_s_next * U_values[h + 1, s_next]
                        mu_W += p_s_next * W_values[h + 1, s_next]

                    var_U, var_W = 0.0, 0.0
                    for s_next in range(S):
                        p_s_next = P_hat[h, s, a, s_next]
                        var_U += p_s_next * (U_values[h + 1, s_next] - mu_U)**2
                        var_W += p_s_next * (W_values[h + 1, s_next] - mu_W)**2

                    var_max = max(var_U, var_W)
                    b_calc = c1 * np.sqrt(var_max * L1 / n) + c2 * (H - h) * L1 / n
                    b = min(b_calc, float(H - h))
                
                U_Q[h, s, a] = min(R[h, s, a] + mu_U + b, float(H - h))
                q_lower_h[s, a] = max(0.0, R[h, s, a] + mu_W - b)

        for s in numba.prange(S):
            max_upper_q = -np.inf
            max_lower_q = -np.inf
            for a in range(A):
                if U_Q[h, s, a] > max_upper_q: max_upper_q = U_Q[h, s, a]
                if q_lower_h[s, a] > max_lower_q: max_lower_q = q_lower_h[s, a]

            U_values[h, s] = max_upper_q if max_upper_q != -np.inf else 0.0
            W_values[h, s] = max_lower_q if max_lower_q != -np.inf else 0.0

    return U_values, W_values, U_Q


def compute_offline_bounds(
    mdp: TabularMDP,
    trajectories: List[Tuple[List[int], List[int], List[float]]],
    delta: float,
    seed: int | None = None,
    use_h_split: bool = True
) -> Dict[str, Any]:
    H, S, A = mdp.H, mdp.S, mdp.A
    K = len(trajectories)
    assert K % H == 0, (
        f"Number of trajectories K={K} must be a multiple of the horizon H={H}.")

    N_sa_arr = np.zeros((H, S, A), dtype=np.int64)
    N_sas_arr = np.zeros((H, S, A, S), dtype=np.int64)

    if use_h_split:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(K)
        buckets: List[List[Tuple[List[int], List[int], List[float]]]] = [[] for _ in range(H)]
        for pos, idx in enumerate(indices):
            h = pos % H
            buckets[h].append(trajectories[idx])
        
        for h in range(H):
            for (states, actions, _) in buckets[h]:
                s, a, s_next = states[h], actions[h], states[h+1]
                N_sa_arr[h, s, a] += 1
                N_sas_arr[h, s, a, s_next] += 1
    else:
        for trajectory in trajectories:
            states, actions, _ = trajectory
            for h in range(H):
                s, a, s_next = states[h], actions[h], states[h+1]
                N_sa_arr[h, s, a] += 1
                N_sas_arr[h, s, a, s_next] += 1

    P_hat_arr = np.zeros((H, S, A, S), dtype=np.float64)
    for h in range(H):
        for s in range(S):
            for a in range(A):
                n = N_sa_arr[h, s, a]
                if n == 0:
                    P_hat_arr[h, s, a, :] = 1.0 / S
                else:
                    P_hat_arr[h, s, a, :] = N_sas_arr[h, s, a, :] / n

    U_values_arr, W_values_arr, U_Q_arr = _jit_offline_bounds_core(
        H, S, A, mdp.R, P_hat_arr, N_sa_arr, delta
    )

    U_values = [U_values_arr[h] for h in range(H)]
    W_values = [W_values_arr[h] for h in range(H)]
    U_Q = [U_Q_arr[h] for h in range(H)]

    M: List[np.ndarray] = [0.5 * (U_values[h] + W_values[h]) for h in range(H)]
    D: List[np.ndarray] = [U_values[h] - W_values[h] for h in range(H)]

    R_next: List[float] = []
    D_next_max: List[float] = []
    for h in range(H):
        if h + 1 < H:
            r_val = float(np.max(U_values_arr[h + 1]) - np.min(W_values_arr[h + 1]))
            d_val = float(np.max(D[h + 1]))
        else:
            r_val, d_val = 0.0, 0.0
        R_next.append(r_val)
        D_next_max.append(d_val)

    return {
        'U_values': U_values, 'W_values': W_values, 'U_Q': U_Q,
        'M': M, 'D': D, 'R_next': R_next, 'D_next_max': D_next_max,
        # Export counts for Count-Based Initialization
        'N_sa': N_sa_arr,
        'N_sas': N_sas_arr
    }