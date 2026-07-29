import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    print("Loading benchmark_data_2d.json...")
    with open("benchmark_data_2d.json", "r") as f:
        results = json.load(f)
        
    ENGINES = ["MPS", "MLX"]
    PRECISIONS = ["FP32", "FP16", "BF16"]
    
    OFFSET = 1024
    SWEEP_RANGE = 64
    M_RANGE = list(range(OFFSET, OFFSET + SWEEP_RANGE))
    N_RANGE = list(range(OFFSET, OFFSET + SWEEP_RANGE))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Matrix Multiplication 2D Sweep Latency (ms)\nInner Dim (K) = 1024, Outer Dims (M, N) = 1024 to 1087\nNormalized Per-Framework (Row)", fontsize=16)
    
    # We will have one colorbar per row (engine)
    cbar_ax_mps = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    cbar_ax_mlx = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    
    for i, engine in enumerate(ENGINES):
        # Calculate row-specific min and max for normalization
        row_vals = []
        for prec in PRECISIONS:
            vals = results.get(engine, {}).get(prec, {}).values()
            row_vals.extend([v for v in vals if v is not None])
            
        if not row_vals:
            print(f"No valid data for {engine}")
            continue
            
        row_min = min(row_vals)
        row_max = max(row_vals)
        print(f"{engine} Normalization: Min = {row_min:.3f} ms, Max = {row_max:.3f} ms")
        
        cbar_ax = cbar_ax_mps if engine == "MPS" else cbar_ax_mlx
        
        for j, prec in enumerate(PRECISIONS):
            ax = axes[i, j]
            
            data_dict = results.get(engine, {}).get(prec, {})
            valid_data = {k: v for k, v in data_dict.items() if v is not None}
            
            if not valid_data:
                ax.text(0.5, 0.5, "Unsupported\nor Failed", ha='center', va='center', transform=ax.transAxes, color="red")
                ax.set_title(f"{engine} | {prec}")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
                
            heat_data = np.full((len(M_RANGE), len(N_RANGE)), np.nan)
            for m_idx, m_val in enumerate(M_RANGE):
                for n_idx, n_val in enumerate(N_RANGE):
                    key = f"{m_val}_{n_val}"
                    if key in valid_data:
                        # Normalize to [-1, 1] relative to THIS ENGINE'S min/max
                        if row_max > row_min:
                            norm_val = 2 * ((valid_data[key] - row_min) / (row_max - row_min)) - 1
                        else:
                            norm_val = 0
                        heat_data[m_idx, n_idx] = norm_val
                        
            sns.heatmap(heat_data, ax=ax, cmap="coolwarm", vmin=-1.0, vmax=1.0,
                        cbar=(j == 2), cbar_ax=cbar_ax if j == 2 else None,
                        yticklabels=[str(m) if m % 8 == 0 else "" for m in M_RANGE], 
                        xticklabels=[str(n) if n % 8 == 0 else "" for n in N_RANGE])
            
            # Add subtitle with absolute speeds for context
            ax.set_title(f"{engine} | {prec}\nMin: {min(valid_data.values()):.3f}ms  Max: {max(valid_data.values()):.3f}ms")
            ax.set_xlabel("N Dimension (Cols of B)")
            if j == 0:
                ax.set_ylabel("M Dimension (Rows of A)")
                
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig("heat_2d_per_row.png", dpi=300)
    print("Saved heat_2d_per_row.png successfully.")

if __name__ == "__main__":
    main()
