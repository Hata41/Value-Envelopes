# Learning Upper–Lower Value Envelopes to Shape Online RL: A Principled Approach

This code generates the data and TikZ figures for the article:
**"Learning Upper–Lower Value Envelopes to Shape Online RL: A Principled Approach"**

## Dependencies

The `requirements.txt` file contains the necessary Python packages to run the scripts. You can install them via:

```bash
pip install -r requirements.txt

```

## Data Generation

The following scripts contain the hyperparameters used in the article and must be run to generate the experimental data:

1. **`generate_R_vs_D.py`**: Generates data regarding the convergence of offline bounds (Figure 2).
* *Output*: `R_vs_D.dat`


2. **`generate_regrets.py`**: Runs the online learning experiments comparing Standard UCBVI, Count-Init UCBVI, V-Shaping, and Q-Shaping for various offline dataset sizes () (Figure 3).
* *Output*: Folders named `RegretCurves_...` containing `.dat` files.


3. **`generate_mdp_trials.py`**: Runs the "Expanding Reward" and "Sliding Window" experiments to calculate relative regret improvement (Figures 4 & 5).
* *Note*: This script is computationally intensive and may take a few hours on 20 CPU cores.
* *Output*: Folders named `ExpandingReward` and `SlidingWindow_width0p10`.



## Figures

The data produced by the scripts above is used by the following LaTeX files to generate the TikZ figures found in the article:

* **Figure 2**: `R_vs_D.tex` (uses `R_vs_D.dat`)
* **Figure 3**: `plot_v_shaping_vs_K.tex` and `plot_q_shaping_vs_K.tex` (uses data from `RegretCurves_...`)
* **Figures 4 & 5**: `mdp_trials.tex` (uses data from `ExpandingReward` and `SlidingWindow_width0p10`)

### Quick Visualization

Python scripts are also provided to generate PNG previews of the results without compiling LaTeX:

* `python_plot_R_vs_D.py`
* `python_plot_regrets.py`
* `python_plot_mdp_trials.py`