import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_1d(data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("1D Sweep Analysis: Isolating Dimensions", fontsize=16)
    
    dims = ["M", "N", "K"]
    for i, dim in enumerate(dims):
        ax = axes[i]
        
        # Plot MPS
        mps_x = sorted([int(k) for k in data["MPS"][dim].keys()])
        mps_y = [data["MPS"][dim][str(x)] for x in mps_x]
        ax.plot(mps_x, mps_y, label="MPS FP16", color="blue", linewidth=1.5)
        
        # Plot MLX
        mlx_x = sorted([int(k) for k in data["MLX"][dim].keys()])
        mlx_y = [data["MLX"][dim][str(x)] for x in mlx_x]
        ax.plot(mlx_x, mlx_y, label="MLX FP16", color="orange", linewidth=1.5)
        
        ax.set_title(f"Sweep {dim} (Other dims = 2048)")
        ax.set_xlabel(f"{dim} Size")
        ax.set_ylabel("Latency (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("advanced_1d.png", dpi=300)
    plt.close()

def plot_2d(data):
    engines = ["MPS", "MLX"]
    axes_list = ["M_vs_N", "M_vs_K", "N_vs_K"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("2D Sweeps: Cache Alignment Interactions", fontsize=16)
    
    for i, engine in enumerate(engines):
        for j, ax_name in enumerate(axes_list):
            ax = axes[i, j]
            grid_data = data[engine][ax_name]
            
            # Extract distinct x and y coordinates
            x_coords = sorted(list(set(int(k.split('_')[0]) for k in grid_data.keys())))
            y_coords = sorted(list(set(int(k.split('_')[1]) for k in grid_data.keys())))
            
            heat = np.full((len(x_coords), len(y_coords)), np.nan)
            for xi, x in enumerate(x_coords):
                for yi, y in enumerate(y_coords):
                    key = f"{x}_{y}"
                    if key in grid_data:
                        heat[xi, yi] = grid_data[key]
                        
            # Normalize row-wise for visibility
            h_min, h_max = np.nanmin(heat), np.nanmax(heat)
            heat_norm = 2 * ((heat - h_min) / (h_max - h_min)) - 1
            
            sns.heatmap(heat_norm, ax=ax, cmap="coolwarm", cbar=True,
                        xticklabels=[str(y) if y % 64 == 0 else "" for y in y_coords],
                        yticklabels=[str(x) if x % 64 == 0 else "" for x in x_coords])
            
            ax.set_title(f"{engine} | {ax_name}\nMin: {h_min:.2f}ms  Max: {h_max:.2f}ms")
            
            if ax_name == "M_vs_N":
                ax.set_xlabel("N")
                ax.set_ylabel("M")
            elif ax_name == "M_vs_K":
                ax.set_xlabel("K")
                ax.set_ylabel("M")
            elif ax_name == "N_vs_K":
                ax.set_xlabel("K")
                ax.set_ylabel("N")
                
    plt.tight_layout()
    plt.savefig("advanced_2d.png", dpi=300)
    plt.close()
    
def plot_determinism(data):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Determinism Analysis: Fast, Median, and Slow Spots (150 Runs Each)", fontsize=16)
    
    engines = ["MPS", "MLX"]
    conditions = ["cold", "median", "hot"]
    
    for i, engine in enumerate(engines):
        for j, cond in enumerate(conditions):
            ax = axes[i, j]
            cond_data = data[engine][cond]
            
            for key, latencies in cond_data.items():
                sns.kdeplot(latencies, ax=ax, alpha=0.5)
                
            ax.set_title(f"{engine} - {cond.capitalize()} 5 Spots Distribution")
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel("Density")
            
    plt.tight_layout()
    plt.savefig("determinism_plot.png", dpi=300)
    plt.close()

def main():
    print("Plotting Advanced Sweeps...")
    with open("advanced_benchmark_data.json", "r") as f:
        data = json.load(f)
        
    plot_1d(data["1d"])
    plot_2d(data["2d"])
    
    try:
        with open("determinism_data_v2.json", "r") as f:
            det_data = json.load(f)
        plot_determinism(det_data)
    except FileNotFoundError:
        print("Determinism v2 data not found. Run run_determinism_v2.py first.")

if __name__ == "__main__":
    main()
