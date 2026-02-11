# Learning Upper–Lower Value Envelopes to Shape Online RL: A Principled Approach

This repository provides a high-performance Rust implementation for the experiments and figures presented in the article:
**"Learning Upper–Lower Value Envelopes to Shape Online RL: A Principled Approach"**

## Prerequisites

- **Rust**: [Install Rust](https://www.rust-lang.org/tools/install)
- **pdflatex**: Required for generating publication-quality PDF plots (included in TeX Live, MiKTeX, etc.)

## Quick Start

### 1. Build the project
```bash
cargo build --release
```

### 2. Run Experiments (Batch Mode)
The recommended way to run experiments is using a YAML configuration file. This allows you to chain multiple runs and specify exact agent configurations.

```bash
cargo run --release -- --config batch.yaml
```

### 3. Run a Single Experiment (CLI Mode)
You can also run experiments directly via CLI arguments:

```bash
cargo run --release -- --experiment regret_curves -H 4 -s 12 -a 3 -t 10000 --n-seeds 10
```

## Configuration Schema

Experiments can be configured via YAML. You can provide a single experiment, a list of experiments, or a list of paths to other YAML files.

### Key Configuration Fields:
- `experiment`: One of `regret_curves`, `expanding_reward`, `sliding_window`, `convergence`.
- `h`, `s`, `a`: MDP dimensions (Horizon, States, Actions).
- `t`: Number of online episodes.
- `n_seeds`: Number of independent runs for averaging.
- `baseline_agents`: List of standard agents (e.g., `standard_hoeffding`, `standard_bernstein`).
- `shaping_agents`: List of agents using offline data (e.g., `v_shaping`, `q_shaping`, `count_init_hoeffding`, `count_init_bernstein`).
- `compile`: Boolean to enable/disable automated LaTeX compilation.

## Available Experiments

| Experiment | Config File | Figure Reference | Output Plots (PDF) |
| :--- | :--- | :--- | :--- |
| **Regret Curves** | `regret_curves.yaml` | Figure 3 | [plot_v_shaping_vs_K.pdf](pdf/plot_v_shaping_vs_K.pdf), [plot_q_shaping_vs_K.pdf](pdf/plot_q_shaping_vs_K.pdf) |
| **Expanding Reward** | `expanding_reward.yaml` | Figure 4 | [expanding_reward_mdp_trials.pdf](pdf/expanding_reward_mdp_trials.pdf) |
| **Sliding Window** | `sliding_window.yaml` | Figure 5 | [sliding_window_mdp_trials.pdf](pdf/sliding_window_mdp_trials.pdf) |
| **Convergence** | `convergence.yaml` | Figure 2 | [R_vs_D.pdf](pdf/R_vs_D.pdf) |
| **Batch Run** | `batch.yaml` | All | All of the above |

## Output Structure

- `data/`: Raw `.dat` files with mean/std results.
- `png/`: Fast-preview plots generated via the Rust `plotters` crate.
- `pdf/`: Publication-quality TikZ/PGFPlots generated via `pdflatex`.

## Simulation Workflow

The implementation follows a modular workflow:
1. **MDP Generation**: Creates tabular MDPs with specified reward/transition structures.
2. **Offline Data**: Generates a dataset of $K$ trajectories to compute Value Envelopes.
3. **Agent Runs**: Executes selected agents in parallel using `rayon`.
4. **Data Processing**: Aggregates results and saves them to the `data/` folder.
5. **Visualization**: Automatically generates LaTeX source, compiles it with `pdflatex`, and moves the resulting PDFs to the `pdf/` folder.

---
*Note: This Rust implementation is optimized for multi-core execution and replaces the legacy Python prototype.*
