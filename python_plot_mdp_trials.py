import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_experiment(folder_path, title, xlabel, output_filename, scale='linear', invert_xaxis=False):
    if not os.path.exists(folder_path):
        print(f"Skipping {folder_path} (Directory not found)")
        return

    # Files to plot
    files_to_plot = [
        {
            "filename": "Bonus_Shaping_Only.dat",
            "label": "Full-Bonus",
            "color": "purple",
            "marker": "s", # square
            "linestyle": "-"
        },
        {
            "filename": "Upper_Bonus_Shaping.dat",
            "label": "Upper-Bonus",
            "color": "green", 
            "marker": "o", # circle
            "linestyle": "-"
        },
        {
            "filename": "Count_Init_UCBVI.dat",
            "label": "Count-Init",
            "color": "blue",
            "marker": "^", # triangle up
            "linestyle": "-"
        }
    ]

    plt.figure(figsize=(10, 7))
    
    has_data = False
    
    for entry in files_to_plot:
        filepath = os.path.join(folder_path, entry["filename"])
        if os.path.exists(filepath):
            try:
                # Read data expecting 3 columns: X, Y (Mean), Std
                df = pd.read_csv(filepath, delim_whitespace=True, comment='#', names=["X", "Y", "Std"])
                
                # Plot Mean
                plt.plot(df["X"], df["Y"], 
                         label=entry["label"], 
                         color=entry["color"], 
                         marker=entry["marker"],
                         linestyle=entry["linestyle"],
                         linewidth=2,
                         markersize=6)
                
                # Plot Std Dev Shading
                if "Std" in df.columns and not df["Std"].isnull().all():
                    plt.fill_between(df["X"], 
                                     df["Y"] - df["Std"], 
                                     df["Y"] + df["Std"], 
                                     color=entry["color"], 
                                     alpha=0.15) # Transparent shading

                has_data = True
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        else:
            print(f"File not found: {filepath}")

    if not has_data:
        plt.close()
        return

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Relative Regret Improvement", fontsize=12)
    
    # Grid and Scales
    plt.grid(True, which="major", linestyle='--', alpha=0.7)
    if scale == 'log':
        plt.xscale('log')
    
    if invert_xaxis:
        plt.gca().invert_xaxis()

    # Format Y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Adjust Y-limits to fit variance if needed, or keep fixed range
    # plt.ylim(-0.2, 1.1) 

    # Legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), 
               ncol=3, frameon=False, fontsize=12)

    # Save
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Saved plot to {output_filename}")
    plt.close()

def main():
    # 1. Expanding Reward
    plot_experiment(
        folder_path="ExpandingReward",
        title="Expanding Reward: Bonus Shaping Comparison",
        xlabel=r"Reward Range Width ($x$) in $R_{\text{term}} = [1-x, 1.0]$",
        output_filename="ExpandingReward_Comparison_Std.png",
        scale="log"
    )

    # 3. Sliding Window
    plot_experiment(
        folder_path="SlidingWindow_width0p10",
        title="Sliding Window: Bonus Shaping Comparison",
        xlabel=r"Reward Window Start Location ($x$) in $R_{\text{term}} = [x, x+0.1]$",
        output_filename="SlidingWindow_Comparison_Std.png",
        scale="linear"
    )

if __name__ == "__main__":
    main()