# Learning Upper–Lower Value Envelopes to Shape Online RL: A Principled Approach

This code generates the data and TikZ figures for the article:
**"Learning Upper–Lower Value Envelopes to Shape Online RL: A Principled Approach"**

## Setup

Ensure you have the Rust toolchain installed.

## Data Generation & Visualization

The Rust implementation handles both data generation and visualization.

### 1. Build the project
```bash
cargo build --release
```

### 2. Run Experiments & Generate Figures

| Experiment | Command | Figure Reference | Output Plot |
| :--- | :--- | :--- | :--- |
| **Regret Curves** | `cargo run --release -- --experiment regret_curves --t 1000000 --n-seeds 8` | Figure 3 | `rust_v_shaping.png`, `rust_q_shaping.png` |
| **Expanding Reward** | `cargo run --release -- --experiment expanding_reward --t 100000 --n-seeds 8` | Figure 4 | `ExpandingReward.png` |
| **Sliding Window** | `cargo run --release -- --experiment sliding_window --t 100000 --n-seeds 8` | Figure 5 | `SlidingWindow_width0p10.png` |
| **Convergence** | `cargo run --release -- --experiment convergence` | Figure 2 | `rust_R_vs_D.png` |

*Note: You can adjust the horizon `h`, states `s`, actions `a`, episodes `t`, and `n_seeds` via CLI arguments.*

## Simulation Details

The experiments generate `.dat` files in specialized folders (e.g., `RegretCurves_...`) which are then automatically processed by the built-in Rust plotting module using the `plotters` crate.

---
*Note: The legacy Python implementation has been fully replaced by this high-performance Rust port.*