import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def parse_folder_name(folder_name):
    """Extracts H, S, A, T from the folder name."""
    pattern = r"H(\d+)_S(\d+)_A(\d+)_T(\d+)"
    match = re.search(pattern, folder_name)
    
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
    else:
        raise ValueError(f"Could not parse parameters from folder name: {folder_name}")

def get_available_k_values(main_folder):
    """Scans the main folder for subdirectories named 'K={number}'."""
    subdirs = glob.glob(os.path.join(main_folder, "K=*"))
    k_values = []
    for path in subdirs:
        dir_name = os.path.basename(path)
        try:
            k_val = int(dir_name.split('=')[1])
            k_values.append(k_val)
        except (IndexError, ValueError):
            continue
    return sorted(k_values)

def plot_regret_comparison(main_folder, shaping_file_name, shaping_label, output_filename, plot_limit=None):
    if not os.path.exists(main_folder):
        print(f"Error: Folder '{main_folder}' not found.")
        return

    try:
        H, S, A, T_total = parse_folder_name(main_folder)
        print(f" inferred -> H={H}, S={S}, A={A}, T={T_total}")
    except ValueError as e:
        print(e)
        return

    plt.figure(figsize=(10, 7))

    # Helper to read file and return DataFrame
    def read_and_filter(filepath):
        try:
            df = pd.read_csv(filepath, delim_whitespace=True, comment='#')
            if plot_limit:
                df = df[df["Episode"] <= plot_limit]
            return df
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
            return None

    # --- Plot Standard UCBVI ---
    base_file = os.path.join(main_folder, "Standard_UCBVI.dat")
    if os.path.exists(base_file):
        df = read_and_filter(base_file)
        if df is not None:
            x_data = df["Episode"] / T_total if plot_limit is None else df["Episode"]
            mean = df["CumulativeRegret"]
            
            plt.plot(x_data, mean, color='black', linestyle='--', linewidth=2, label='Standard UCBVI')
            
            # Plot Std Dev if available
            if "StdDev" in df.columns:
                std = df["StdDev"]
                plt.fill_between(x_data, mean - std, mean + std, color='black', alpha=0.1)
    else:
        print(f"  Warning: Baseline file not found.")

    # --- Plot Experiments ---
    k_values = get_available_k_values(main_folder)
    if not k_values:
        print("  Warning: No K=... subdirectories found.")
        
    cmap = plt.get_cmap('coolwarm_r')
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(k_values))]

    for k, color in zip(k_values, colors):
        # 1. Shaping Method
        filename = os.path.join(main_folder, f"K={k}", shaping_file_name)
        if os.path.exists(filename):
            df = read_and_filter(filename)
            if df is not None:
                x_data = df["Episode"] / T_total if plot_limit is None else df["Episode"]
                mean = df["CumulativeRegret"]
                
                plt.plot(x_data, mean, color=color, marker='s', linestyle='-', linewidth=1.5, 
                         markersize=5, markevery=0.1, label=f'{shaping_label} (K={k})')
                
                if "StdDev" in df.columns:
                    std = df["StdDev"]
                    plt.fill_between(x_data, mean - std, mean + std, color=color, alpha=0.15)

        # 2. Count-Init Method
        filename_count = os.path.join(main_folder, f"K={k}", "Count_Init_UCBVI.dat")
        if os.path.exists(filename_count):
            df = read_and_filter(filename_count)
            if df is not None:
                x_data = df["Episode"] / T_total if plot_limit is None else df["Episode"]
                mean = df["CumulativeRegret"]
                
                plt.plot(x_data, mean, color=color, marker='o', linestyle='--', linewidth=1.5, 
                         markersize=5, markevery=0.1, alpha=0.6, label=f'Count-Init (K={k})')
                
                if "StdDev" in df.columns:
                    std = df["StdDev"]
                    plt.fill_between(x_data, mean - std, mean + std, color=color, alpha=0.05)

    # Formatting
    title_suffix = f" (First {plot_limit} eps)" if plot_limit else f" (Full T={T_total})"
    plt.title(f"{shaping_label} vs Offline Size (K) with StdDev\n{title_suffix}", fontsize=14)
    
    if plot_limit:
        plt.xlabel("Episode Number", fontsize=12)
    else:
        plt.xlabel(f"Fraction of T (T={T_total})", fontsize=12)
        plt.xlim(0, 1.0)

    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.grid(True, which="major", linestyle='-', alpha=0.6)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(loc='upper left', frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")
    plt.close()

def main():
    # CHANGE THIS FOLDER NAME TO MATCH YOUR NEW EXPERIMENT FOLDER
    # Note: The folder name might have changed if you ran T=1e6 vs T=1e7
    target_folder = "RegretCurves_H4_S12_A3_T1000000_Layered" 
    
    plot_regret_comparison(target_folder, "V_Shaping.dat", "V-Shaping", "python_v_shaping_std.png")
    plot_regret_comparison(target_folder, "Q_Shaping.dat", "Q-Shaping", "python_q_shaping_std.png")

if __name__ == "__main__":
    main()