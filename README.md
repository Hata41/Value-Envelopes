This code generates the data and tikz figures for the article :
"Learning Upper–Lower Value Envelopes to Shape Online RL: A Principled Approach"

The requirements.txt contains the necessary python packages to run the scripts.

The scripts to be run (containing the hyperparameters used in the article) are the scripts : 
main.py, mdp_trials.py and width_plot.py
The mdp_trials.py takes a few hours on 20 CPU cores.
These scripts produce the data necessary for 
plot_q_shaping_vs_K.tex
plot_v_shaping_vs_K.tex
width_plot.tex
compile_plots.tex
to produce the tikz figures present in the article.
A copy of the said data and figures can be found in the article_data folder.