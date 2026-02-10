import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    filename = "R_vs_D.dat"
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run width_plot.py first.")
        return

    try:
        # Read the space-separated data
        df = pd.read_csv(filename, delim_whitespace=True)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    plt.figure(figsize=(10, 7))

    # 1. Plot Local Widths (D_s columns)
    # The tex file plotted specific states, here we plot all D_s columns found
    d_columns = [col for col in df.columns if col.startswith('D_s')]
    
    first_d = True
    for col in d_columns:
        label = r"$D_h(s)$ (Local Widths)" if first_d else None
        plt.plot(df["K"], df[col], 
                 color='orange', 
                 marker='s', 
                 markersize=4, 
                 linewidth=1, 
                 label=label,
                 alpha=0.8)
        first_d = False

    # 2. Plot In-Layer Range (R_h_layer)
    if "R_h_layer" in df.columns:
        plt.plot(df["K"], df["R_h_layer"], 
                 color='blue', 
                 marker='o', 
                 markersize=6, 
                 linewidth=2, 
                 label=r"$R_h$ (In-Layer Range)")

    # 3. Plot Optimal Range (span_h_layer)
    if "span_h_layer" in df.columns:
        plt.plot(df["K"], df["span_h_layer"], 
                 color='red', 
                 linestyle='--', 
                 linewidth=2, 
                 label=r"$\mathrm{Range}(V_h^*)$")

    # Formatting
    plt.title("Convergence of Bounds at h=2", fontsize=14)
    plt.xlabel("Offline Dataset Size (K trajectories)", fontsize=12)
    plt.ylabel("Width of Bounding Interval", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Limits (mimicking the tikz defaults usually means starting at 0)
    plt.ylim(bottom=0)
    plt.xlim(left=0)

    plt.legend(loc='upper right', frameon=False, fontsize=12)

    output_file = "R_vs_D.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")
    plt.close()

if __name__ == "__main__":
    main()