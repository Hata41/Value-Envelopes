from __future__ import annotations

import numpy as np
import numba
from typing import List, Tuple, Dict, Any

from environments import TabularMDP
from utils import value_iteration, _jit_evaluate_policy_core


@numba.jit(nopython=True, cache=True)
def _jit_run_v_shaping_core(
    H: int, S: int, A: int, R: np.ndarray, P_true: np.ndarray, rho: np.ndarray,
    T: int, delta: float, opt_global: float,
    U_values: np.ndarray, M: np.ndarray, D: np.ndarray, R_next: np.ndarray
):
    N_sa = np.zeros((H, S, A), dtype=np.int64)
    N_sas = np.zeros((H, S, A, S), dtype=np.int64)

    L = np.log((8.0 * S * A * H * T) / max(delta, 1e-12))
    c1, c2 = 2.0, 14.0 / 3.0

    regrets = np.zeros(T, dtype=np.float64)
    rewards = np.zeros(T, dtype=np.float64)

    for t in range(T):
        P_hat = np.zeros((H, S, A, S), dtype=np.float64)
        for h_ in range(H):
            for s_ in range(S):
                for a_ in range(A):
                    n = N_sa[h_, s_, a_]
                    if n == 0:
                        P_hat[h_, s_, a_, :] = 1.0 / S
                    else:
                        P_hat[h_, s_, a_, :] = N_sas[h_, s_, a_, :] / n

        Q_hat = np.zeros((H, S, A), dtype=np.float64)
        V_hat = np.zeros((H + 1, S), dtype=np.float64)
        for h in range(H - 1, -1, -1):
            for s in range(S):
                for a in range(A):
                    n = N_sa[h, s, a]
                    b = 0.0
                    if n <= 1:
                        b = R_next[h]
                    else:
                        M_next_h = M[h + 1] if (h + 1 < H) else np.zeros(S)
                        D_next_h = D[h + 1] if (h + 1 < H) else np.zeros(S)
                        
                        mu_M = np.dot(P_hat[h, s, a, :], M_next_h)
                        var_M = np.dot(P_hat[h, s, a, :], (M_next_h - mu_M)**2)
                        E_D2 = np.dot(P_hat[h, s, a, :], D_next_h**2)
                        
                        sigma = np.sqrt(max(var_M, 0.0)) + 0.5 * np.sqrt(max(E_D2, 0.0))
                        b_calc = c1 * sigma * np.sqrt(L / n) + c2 * R_next[h] * (L / n)
                        b = min(b_calc, R_next[h])
                    
                    expected_v = np.dot(P_hat[h, s, a, :], V_hat[h + 1, :])
                    Q_hat[h, s, a] = R[h, s, a] + expected_v + b

            for s in range(S):
                v_val = np.max(Q_hat[h, s, :])
                V_hat[h, s] = min(v_val, U_values[h, s]) # V-Shaping Clip
                
        policy = np.zeros((H, S), dtype=np.int64)
        for h_ in range(H):
            for s_ in range(S):
                policy[h_, s_] = np.argmax(Q_hat[h_, s_, :])
        
        V_pi = _jit_evaluate_policy_core(H, S, A, R, P_true, policy)
        expected_return = np.dot(rho, V_pi[0, :])
        regrets[t] = opt_global - expected_return

        s_current = np.searchsorted(np.cumsum(rho), np.random.rand())
        ep_reward = 0.0
        for h in range(H):
            a_current = policy[h, s_current]
            s_next = np.searchsorted(np.cumsum(P_true[h, s_current, a_current, :]), np.random.rand())
            
            N_sa[h, s_current, a_current] += 1
            N_sas[h, s_current, a_current, s_next] += 1
            ep_reward += R[h, s_current, a_current]
            s_current = s_next
        rewards[t] = ep_reward

    return regrets, rewards


@numba.jit(nopython=True, cache=True)
def _jit_run_q_shaping_core(
    H: int, S: int, A: int, R: np.ndarray, P_true: np.ndarray, rho: np.ndarray,
    T: int, delta: float, opt_global: float,
    U_Q: np.ndarray, M: np.ndarray, D: np.ndarray, R_next: np.ndarray
):
    """JIT-compiled core for the entire QShapingAgent learning process."""
    N_sa = np.zeros((H, S, A), dtype=np.int64)
    N_sas = np.zeros((H, S, A, S), dtype=np.int64)

    L = np.log((4.0 * S * A * H * T) / max(delta, 1e-12))
    c1, c2 = 2.0, 14.0 / 3.0

    regrets = np.zeros(T, dtype=np.float64)
    rewards = np.zeros(T, dtype=np.float64)

    for t in range(T):
        P_hat = np.zeros((H, S, A, S), dtype=np.float64)
        for h_ in range(H):
            for s_ in range(S):
                for a_ in range(A):
                    n = N_sa[h_, s_, a_]
                    if n == 0:
                        P_hat[h_, s_, a_, :] = 1.0 / S
                    else:
                        P_hat[h_, s_, a_, :] = N_sas[h_, s_, a_, :] / n

        Q_hat = np.zeros((H, S, A), dtype=np.float64)
        V_hat = np.zeros((H + 1, S), dtype=np.float64)
        for h in range(H - 1, -1, -1):
            for s in range(S):
                for a in range(A):
                    n = N_sa[h, s, a]
                    b = 0.0
                    if n <= 1:
                        b = R_next[h]
                    else:
                        M_next_h = M[h + 1] if (h + 1 < H) else np.zeros(S)
                        D_next_h = D[h + 1] if (h + 1 < H) else np.zeros(S)
                        
                        mu_M = np.dot(P_hat[h, s, a, :], M_next_h)
                        var_M = np.dot(P_hat[h, s, a, :], (M_next_h - mu_M)**2)
                        E_D2 = np.dot(P_hat[h, s, a, :], D_next_h**2)
                        
                        sigma = np.sqrt(max(var_M, 0.0)) + 0.5 * np.sqrt(max(E_D2, 0.0))
                        b_calc = c1 * sigma * np.sqrt(L / n) + c2 * R_next[h] * (L / n)
                        b = min(b_calc, R_next[h])
                    
                    expected_v = np.dot(P_hat[h, s, a, :], V_hat[h + 1, :])
                    q_val_raw = R[h, s, a] + expected_v + b
                    Q_hat[h, s, a] = min(q_val_raw, U_Q[h, s, a]) # Q-Shaping Clip

            for s in range(S):
                V_hat[h, s] = np.max(Q_hat[h, s, :])

        policy = np.zeros((H, S), dtype=np.int64)
        for h_ in range(H):
            for s_ in range(S):
                policy[h_, s_] = np.argmax(Q_hat[h_, s_, :])
        
        V_pi = _jit_evaluate_policy_core(H, S, A, R, P_true, policy)
        expected_return = np.dot(rho, V_pi[0, :])
        regrets[t] = opt_global - expected_return

        s_current = np.searchsorted(np.cumsum(rho), np.random.rand())
        ep_reward = 0.0
        for h in range(H):
            a_current = policy[h, s_current]
            s_next = np.searchsorted(np.cumsum(P_true[h, s_current, a_current, :]), np.random.rand())
            
            N_sa[h, s_current, a_current] += 1
            N_sas[h, s_current, a_current, s_next] += 1
            ep_reward += R[h, s_current, a_current]
            s_current = s_next
        rewards[t] = ep_reward

    return regrets, rewards


@numba.jit(nopython=True, cache=True)
def _jit_run_standard_ucbvi_core(
    H: int, S: int, A: int, R: np.ndarray, P_true: np.ndarray, rho: np.ndarray,
    T: int, delta: float, opt_global: float
):

    N_sa = np.zeros((H, S, A), dtype=np.int64)
    N_sas = np.zeros((H, S, A, S), dtype=np.int64)

    L = np.log((4.0 * S * A * H * T) / max(delta, 1e-12))
    c1, c2 = 2.0, 14.0 / 3.0
    
    R_next_base = np.array([float(H - (h + 1)) if h + 1 < H else 0.0 for h in range(H)])

    regrets = np.zeros(T, dtype=np.float64)
    rewards = np.zeros(T, dtype=np.float64)

    for t in range(T):
        P_hat = np.zeros((H, S, A, S), dtype=np.float64)
        for h_ in range(H):
            for s_ in range(S):
                for a_ in range(A):
                    n = N_sa[h_, s_, a_]
                    if n == 0:
                        P_hat[h_, s_, a_, :] = 1.0 / S
                    else:
                        P_hat[h_, s_, a_, :] = N_sas[h_, s_, a_, :] / n

        Q_hat = np.zeros((H, S, A), dtype=np.float64)
        V_hat = np.zeros((H + 1, S), dtype=np.float64)
        for h in range(H - 1, -1, -1):
            for s in range(S):
                for a in range(A):
                    n = N_sa[h, s, a]
                    b = 0.0
                    if n <= 1:
                        b = R_next_base[h]
                    else:
                        sigma = 0.5 * R_next_base[h] # Hoeffding Style UCBVI as in Gutpa et al. 2022
                        b_calc = c1 * sigma * np.sqrt(L / n) + c2 * R_next_base[h] * (L / n)
                        b = min(b_calc, R_next_base[h])
                    
                    expected_v = np.dot(P_hat[h, s, a, :], V_hat[h + 1, :])
                    Q_hat[h, s, a] = R[h, s, a] + expected_v + b
            
            for s in range(S):
                v_val = np.max(Q_hat[h, s, :])
                V_hat[h, s] = min(v_val, float(H - h))

        policy = np.zeros((H, S), dtype=np.int64)
        for h_ in range(H):
            for s_ in range(S):
                policy[h_, s_] = np.argmax(Q_hat[h_, s_, :])
        
        V_pi = _jit_evaluate_policy_core(H, S, A, R, P_true, policy)
        expected_return = np.dot(rho, V_pi[0, :])
        regrets[t] = opt_global - expected_return

        s_current = np.searchsorted(np.cumsum(rho), np.random.rand())
        ep_reward = 0.0
        for h in range(H):
            a_current = policy[h, s_current]
            s_next = np.searchsorted(np.cumsum(P_true[h, s_current, a_current, :]), np.random.rand())
            
            N_sa[h, s_current, a_current] += 1
            N_sas[h, s_current, a_current, s_next] += 1
            ep_reward += R[h, s_current, a_current]
            s_current = s_next
        rewards[t] = ep_reward

    return regrets, rewards

@numba.jit(nopython=True, cache=True)
def _jit_run_count_initialized_ucbvi_core(
    H: int, S: int, A: int, R: np.ndarray, P_true: np.ndarray, rho: np.ndarray,
    T: int, delta: float, opt_global: float,
    init_N_sa: np.ndarray, init_N_sas: np.ndarray
):
    # Initialize counts with offline data
    N_sa = init_N_sa.copy()
    N_sas = init_N_sas.copy()

    L = np.log((4.0 * S * A * H * T) / max(delta, 1e-12))
    c1, c2 = 2.0, 14.0 / 3.0
    
    R_next_base = np.array([float(H - (h + 1)) if h + 1 < H else 0.0 for h in range(H)])

    regrets = np.zeros(T, dtype=np.float64)
    rewards = np.zeros(T, dtype=np.float64)

    for t in range(T):
        P_hat = np.zeros((H, S, A, S), dtype=np.float64)
        for h_ in range(H):
            for s_ in range(S):
                for a_ in range(A):
                    n = N_sa[h_, s_, a_]
                    if n == 0:
                        P_hat[h_, s_, a_, :] = 1.0 / S
                    else:
                        P_hat[h_, s_, a_, :] = N_sas[h_, s_, a_, :] / n

        Q_hat = np.zeros((H, S, A), dtype=np.float64)
        V_hat = np.zeros((H + 1, S), dtype=np.float64)
        for h in range(H - 1, -1, -1):
            for s in range(S):
                for a in range(A):
                    n = N_sa[h, s, a]
                    b = 0.0
                    if n <= 1:
                        b = R_next_base[h]
                    else:
                        sigma = 0.5 * R_next_base[h] # Hoeffding Style UCBVI as in Gutpa et al. 2022
                        b_calc = c1 * sigma * np.sqrt(L / n) + c2 * R_next_base[h] * (L / n)
                        b = min(b_calc, R_next_base[h])
                    
                    expected_v = np.dot(P_hat[h, s, a, :], V_hat[h + 1, :])
                    Q_hat[h, s, a] = R[h, s, a] + expected_v + b
            
            for s in range(S):
                v_val = np.max(Q_hat[h, s, :])
                V_hat[h, s] = min(v_val, float(H - h))

        policy = np.zeros((H, S), dtype=np.int64)
        for h_ in range(H):
            for s_ in range(S):
                policy[h_, s_] = np.argmax(Q_hat[h_, s_, :])
        
        V_pi = _jit_evaluate_policy_core(H, S, A, R, P_true, policy)
        expected_return = np.dot(rho, V_pi[0, :])
        regrets[t] = opt_global - expected_return

        s_current = np.searchsorted(np.cumsum(rho), np.random.rand())
        ep_reward = 0.0
        for h in range(H):
            a_current = policy[h, s_current]
            s_next = np.searchsorted(np.cumsum(P_true[h, s_current, a_current, :]), np.random.rand())
            
            N_sa[h, s_current, a_current] += 1
            N_sas[h, s_current, a_current, s_next] += 1
            ep_reward += R[h, s_current, a_current]
            s_current = s_next
        rewards[t] = ep_reward

    return regrets, rewards

class StandardUCBVI:
    def __init__(self, mdp: TabularMDP, T: int, delta: float) -> None:
        self.mdp = mdp
        self.T = int(T)
        self.delta = float(delta)
        self.H, self.S, self.A = mdp.H, mdp.S, mdp.A
        V_opt, _ = value_iteration(mdp)
        self.opt_global = float(np.dot(mdp.rho, V_opt[0]))

    def run(self) -> Tuple[List[float], List[float]]:
        regrets_arr, rewards_arr = _jit_run_standard_ucbvi_core(
            self.H, self.S, self.A, self.mdp.R, self.mdp.P, self.mdp.rho,
            self.T, self.delta, self.opt_global
        )
        return regrets_arr.tolist(), rewards_arr.tolist()

class CountInitializedUCBVI:
    def __init__(self, mdp: TabularMDP, offline_bounds: Dict[str, Any], T: int, delta: float) -> None:
        self.mdp = mdp
        self.T = int(T)
        self.delta = float(delta)
        self.H, self.S, self.A = mdp.H, mdp.S, mdp.A
        V_opt, _ = value_iteration(mdp)
        self.opt_global = float(np.dot(mdp.rho, V_opt[0]))
        
        # Load offline counts
        self.init_N_sa = offline_bounds['N_sa']
        self.init_N_sas = offline_bounds['N_sas']

    def run(self) -> Tuple[List[float], List[float]]:
        regrets_arr, rewards_arr = _jit_run_count_initialized_ucbvi_core(
            self.H, self.S, self.A, self.mdp.R, self.mdp.P, self.mdp.rho,
            self.T, self.delta, self.opt_global,
            self.init_N_sa, self.init_N_sas
        )
        return regrets_arr.tolist(), rewards_arr.tolist()

class VShapingAgent:
    def __init__(
        self,
        mdp: TabularMDP,
        offline_bounds: Dict[str, Any],
        T: int,
        delta: float,
    ) -> None:
        self.mdp = mdp
        self.T = int(T)
        self.delta = float(delta)
        self.H, self.S, self.A = mdp.H, mdp.S, mdp.A

        V_opt, _ = value_iteration(mdp)
        self.opt_global = float(np.dot(mdp.rho, V_opt[0]))

        self.U_values_arr = np.stack(offline_bounds['U_values'])
        self.M_arr = np.stack(offline_bounds['M'])
        self.D_arr = np.stack(offline_bounds['D'])
        self.R_next_arr = np.array(offline_bounds['R_next'], dtype=np.float64)

    def run(self) -> Tuple[List[float], List[float]]:
        regrets_arr, rewards_arr = _jit_run_v_shaping_core(
            self.H, self.S, self.A, self.mdp.R, self.mdp.P, self.mdp.rho,
            self.T, self.delta, self.opt_global,
            self.U_values_arr, self.M_arr, self.D_arr, self.R_next_arr
        )
        return regrets_arr.tolist(), rewards_arr.tolist()


class QShapingAgent:
    def __init__(
        self,
        mdp: TabularMDP,
        offline_bounds: Dict[str, Any],
        T: int,
        delta: float,
    ) -> None:
        self.mdp = mdp
        self.T = int(T)
        self.delta = float(delta)
        self.H, self.S, self.A = mdp.H, mdp.S, mdp.A

        V_opt, _ = value_iteration(mdp)
        self.opt_global = float(np.dot(mdp.rho, V_opt[0]))
        
        self.U_Q_arr = np.stack(offline_bounds['U_Q'])
        self.M_arr = np.stack(offline_bounds['M'])
        self.D_arr = np.stack(offline_bounds['D'])
        self.R_next_arr = np.array(offline_bounds['R_next'], dtype=np.float64)

    def run(self) -> Tuple[List[float], List[float]]:
        regrets_arr, rewards_arr = _jit_run_q_shaping_core(
            self.H, self.S, self.A, self.mdp.R, self.mdp.P, self.mdp.rho,
            self.T, self.delta, self.opt_global,
            self.U_Q_arr, self.M_arr, self.D_arr, self.R_next_arr
        )
        return regrets_arr.tolist(), rewards_arr.tolist()
    
class BonusShapingOnlyAgent(VShapingAgent):
    def run(self) -> Tuple[List[float], List[float]]:
        U_values_no_clip = np.full_like(self.U_values_arr, 2*(self.H+1))

        regrets_arr, rewards_arr = _jit_run_v_shaping_core(
            self.H, self.S, self.A, self.mdp.R, self.mdp.P, self.mdp.rho,
            self.T, self.delta, self.opt_global,
            U_values_no_clip, self.M_arr, self.D_arr, self.R_next_arr
        )
        return regrets_arr.tolist(), rewards_arr.tolist()


class UpperBonusShapingAgent(VShapingAgent):
    def __init__(
        self,
        mdp: TabularMDP,
        offline_bounds: Dict[str, Any],
        T: int,
        delta: float,
    ) -> None:
        super().__init__(mdp, offline_bounds, T, delta)
        
        U_values = offline_bounds['U_values']
        W_values_zero = [np.zeros_like(u) for u in U_values]
        
        self.M_arr = np.stack([0.5 * (U_values[h] + W_values_zero[h]) for h in range(self.H)])
        self.D_arr = np.stack([U_values[h] - W_values_zero[h] for h in range(self.H)])
        
        R_next_list = []
        for h in range(self.H):
            if h + 1 < self.H:
                r_val = float(np.max(U_values[h + 1]) - 0.0)
            else:
                r_val = 0.0
            R_next_list.append(r_val)
        self.R_next_arr = np.array(R_next_list, dtype=np.float64)

    def run(self) -> Tuple[List[float], List[float]]:
        U_values_no_clip = np.full_like(self.U_values_arr, 2*(self.H+1))

        regrets_arr, rewards_arr = _jit_run_v_shaping_core(
            self.H, self.S, self.A, self.mdp.R, self.mdp.P, self.mdp.rho,
            self.T, self.delta, self.opt_global,
            U_values_no_clip, self.M_arr, self.D_arr, self.R_next_arr
        )
        return regrets_arr.tolist(), rewards_arr.tolist()
    
    
class UpperVShapingAgent(VShapingAgent):
    def __init__(
        self,
        mdp: TabularMDP,
        offline_bounds: Dict[str, Any],
        T: int,
        delta: float,
    ) -> None:
        super().__init__(mdp, offline_bounds, T, delta)
        
        U_values = offline_bounds['U_values']
        W_values_zero = [np.zeros_like(u) for u in U_values]
        
        self.M_arr = np.stack([0.5 * (U_values[h] + W_values_zero[h]) for h in range(self.H)])
        self.D_arr = np.stack([U_values[h] - W_values_zero[h] for h in range(self.H)])
        
        R_next_list = []
        for h in range(self.H):
            if h + 1 < self.H:
                r_val = float(np.max(U_values[h + 1]) - np.min(W_values_zero[h + 1]))
            else:
                r_val = 0.0
            R_next_list.append(r_val)
        self.R_next_arr = np.array(R_next_list, dtype=np.float64)

    def run(self) -> Tuple[List[float], List[float]]:
        regrets_arr, rewards_arr = _jit_run_v_shaping_core(
            self.H, self.S, self.A, self.mdp.R, self.mdp.P, self.mdp.rho,
            self.T, self.delta, self.opt_global,
            self.U_values_arr, self.M_arr, self.D_arr, self.R_next_arr
        )
        return regrets_arr.tolist(), rewards_arr.tolist()


class UpperQShapingAgent(QShapingAgent):
    def __init__(
        self,
        mdp: TabularMDP,
        offline_bounds: Dict[str, Any],
        T: int,
        delta: float,
    ) -> None:
        super().__init__(mdp, offline_bounds, T, delta)
        
        U_values = offline_bounds['U_values']
        W_values_zero = [np.zeros_like(u) for u in U_values]
        
        self.M_arr = np.stack([0.5 * (U_values[h] + W_values_zero[h]) for h in range(self.H)])
        self.D_arr = np.stack([U_values[h] - W_values_zero[h] for h in range(self.H)])
        
        R_next_list = []
        for h in range(self.H):
            if h + 1 < self.H:
                r_val = float(np.max(U_values[h + 1]) - np.min(W_values_zero[h + 1]))
            else:
                r_val = 0.0
            R_next_list.append(r_val)
        self.R_next_arr = np.array(R_next_list, dtype=np.float64)

    def run(self) -> Tuple[List[float], List[float]]:
        regrets_arr, rewards_arr = _jit_run_q_shaping_core(
            self.H, self.S, self.A, self.mdp.R, self.mdp.P, self.mdp.rho,
            self.T, self.delta, self.opt_global,
            self.U_Q_arr, self.M_arr, self.D_arr, self.R_next_arr
        )
        return regrets_arr.tolist(), rewards_arr.tolist()