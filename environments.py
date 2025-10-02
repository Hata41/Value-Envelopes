from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

class TabularMDP:
    def __init__(self, H: int, S: int, A: int,
                 P: np.ndarray, R: np.ndarray,
                 rho: Optional[np.ndarray] = None) -> None:
        assert P.shape == (H, S, A, S), f"P must have shape (H,S,A,S); got {P.shape}"
        assert R.shape == (H, S, A), f"R must have shape (H,S,A); got {R.shape}"
        self.H, self.S, self.A = int(H), int(S), int(A)
        P_norm = np.clip(P, 0.0, None).astype(float)
        for h in range(H):
            for s in range(S):
                for a in range(A):
                    row, ssum = P_norm[h, s, a], P_norm[h, s, a].sum()
                    if ssum <= 0: P_norm[h, s, a] = 1.0 / float(S)
                    else: P_norm[h, s, a] = row / ssum
        self.P, self.R = P_norm, np.clip(R, 0.0, 1.0).astype(float)
        if rho is None:
            self.rho = np.full(self.S, 1.0 / float(self.S), dtype=float)
        else:
            assert rho.shape == (S,)
            rho, total = np.clip(rho, 0.0, None), float(np.sum(rho))
            self.rho = np.full(self.S, 1.0 / float(self.S), dtype=float) if total <= 0 else (rho / total)
        self._current_state: Optional[int] = None
        self._current_step: Optional[int] = None

    def reset(self) -> int:
        self._current_step = 0
        self._current_state = int(np.random.choice(self.S, p=self.rho))
        return self._current_state

    def step(self, action: int) -> tuple[int, float]:
        assert self._current_state is not None and self._current_step is not None, "Env must be reset"
        h, s = self._current_step, self._current_state
        assert 0 <= h < self.H, "Cannot step after terminal"
        reward = float(self.R[h, s, action])
        next_state = int(np.random.choice(self.S, p=self.P[h, s, action]))
        self._current_state, self._current_step = next_state, h + 1
        return next_state, reward

    @staticmethod
    def random(H: int, S: int, A: int,
               reward_sparse: bool = False,
               layered: bool = False,
               intermediate_reward_range: Optional[Tuple[float, float]] = (0.0, 1.0),
               terminal_reward_range: Optional[Tuple[float, float]] = (0.0, 1.0),
               seed: Optional[int] = 42) -> "TabularMDP":

        rng = np.random.default_rng(seed)
        rho = None

        if layered:
            if S % H != 0: raise ValueError(f"For layered MDP, S({S}) must be divisible by H({H}).")
            S_layer = S // H
            P = np.zeros((H, S, A, S), dtype=float)
            for h in range(H):
                current_layer = range(h * S_layer, (h + 1) * S_layer)
                next_layer_start = ((h + 1) % H) * S_layer
                for s in current_layer:
                    for a in range(A):
                        rand_probs = rng.random(size=S_layer)
                        P[h, s, a, next_layer_start : next_layer_start + S_layer] = rand_probs / rand_probs.sum()
            rho = np.zeros(S, dtype=float)
            rho[0:S_layer] = 1.0 / S_layer
        else:
            P = rng.random(size=(H, S, A, S))

        if intermediate_reward_range is not None:
            if not (isinstance(intermediate_reward_range, (list, tuple)) and len(intermediate_reward_range) == 2):
                raise ValueError("intermediate_reward_range must be a list/tuple of length 2.")
            lower, upper = intermediate_reward_range
            if not (0.0 <= lower <= upper <= 1.0):
                raise ValueError(f"Invalid intermediate_reward_range [{lower}, {upper}]. Must be in [0, 1].")
            
            span = upper - lower
            R = lower + rng.random(size=(H, S, A)) * span
        
        elif not reward_sparse:
            R = rng.random(size=(H, S, A))
        
        else:
            R = np.zeros((H, S, A), dtype=float)

        if terminal_reward_range is not None:
            if not (isinstance(terminal_reward_range, (list, tuple)) and len(terminal_reward_range) == 2):
                raise ValueError("terminal_reward_range must be a list/tuple of length 2.")
            lower, upper = terminal_reward_range
            if not (0.0 <= lower <= upper <= 1.0):
                raise ValueError(f"Invalid terminal_reward_range [{lower}, {upper}]. Must be in [0, 1].")

            span = upper - lower
            v_h_star_values = lower + rng.random(size=S) * span
            
            min_idx, max_idx = rng.choice(S, 2, replace=False)
            v_h_star_values[min_idx] = lower
            v_h_star_values[max_idx] = upper

            R[H - 1, :, :] = v_h_star_values[:, np.newaxis]

        elif reward_sparse:
            goal = rng.integers(0, S)
            if layered:
                goal_layer_start = (H - 1) * (S // H)
                goal = rng.integers(low=goal_layer_start, high=S)
            R[H - 1, goal, :] = 1.0
            
        return TabularMDP(H=H, S=S, A=A, P=P, R=R, rho=rho)