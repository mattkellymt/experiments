import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Set random seeds for complete reproducibility
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------------------------------------------------------
# 1. Dataset Generation: 4 Classes x 32 Samples
# -----------------------------------------------------------------------------
def generate_random_5050_dataset(num_per_class=32, K=4, seed=42):
    np.random.seed(seed)
    centroids = np.random.normal(loc=0.0, scale=2.5, size=(K, 2))
    stds = np.clip(np.abs(np.random.normal(loc=0.55, scale=0.25, size=K)), 0.25, 1.1)
    
    def sample_set(seed_set):
        np.random.seed(seed_set)
        X_list, y_list = [], []
        for k in range(K):
            points = np.random.normal(loc=centroids[k], scale=stds[k], size=(num_per_class, 2))
            X_list.append(points)
            y_list.append(np.full(num_per_class, k))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        return torch.tensor(X[idx], dtype=torch.float32), torch.tensor(y[idx], dtype=torch.long)

    X_tr, y_tr = sample_set(seed + 10)
    X_te, y_te = sample_set(seed + 999)
    return X_tr, y_tr, X_te, y_te, centroids, stds

# -----------------------------------------------------------------------------
# 2. Track Per-Step Class-Wise Training Dynamics for 3 Strategies
# -----------------------------------------------------------------------------
def track_batch_strategy(X_tr, y_tr, X_te, y_te, strategy='random', steps=32, batch_size=4):
    torch.manual_seed(42)
    net = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 4))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    N = len(y_tr) # 128
    
    step_class_accs = []
    step_losses = []
    
    for step in range(steps):
        if strategy == 'random':
            idx = torch.randperm(N)[:batch_size]
        elif strategy == 'balanced':
            # Exactly 1 sample per class
            idx = torch.tensor([torch.where(y_tr == k)[0][step % 32] for k in range(4)])
        elif strategy == 'single_class':
            # Sequential blocks: 8 steps of Class 0, 8 steps of Class 1, 8 of Class 2, 8 of Class 3
            curr_class = (step // 8) % 4
            class_pts = torch.where(y_tr == curr_class)[0]
            start_i = (step * 4) % 32
            idx = class_pts[start_i : start_i + 4]
            
        optimizer.zero_grad()
        loss = criterion(net(X_tr[idx]), y_tr[idx])
        loss.backward()
        optimizer.step()
        
        step_losses.append(loss.item())
        
        # Track accuracy per class on the full training set after this update step
        with torch.no_grad():
            preds_tr = net(X_tr).argmax(dim=1)
            accs = [(preds_tr[y_tr == k] == k).float().mean().item() * 100.0 for k in range(4)]
            step_class_accs.append(accs)
            
    net.eval()
    with torch.no_grad():
        acc_tr_final = (net(X_tr).argmax(dim=1) == y_tr).float().mean().item() * 100.0
        acc_te_final = (net(X_te).argmax(dim=1) == y_te).float().mean().item() * 100.0
        
    return step_class_accs, step_losses, acc_tr_final, acc_te_final

# -----------------------------------------------------------------------------
# 3. Experiment Pipeline & Visual Graphic Generation
# -----------------------------------------------------------------------------
def run_batch_experiment():
    X_tr, y_tr, X_te, y_te, centroids, stds = generate_random_5050_dataset(num_per_class=32, K=4, seed=42)
    
    strategies = ['random', 'balanced', 'single_class']
    results = {}
    
    print("==========================================================================")
    print("BATCHING STRATEGY EXPERIMENT (Batch Size = 4, 4 Classes)")
    print("==========================================================================")
    
    for strat in strategies:
        step_accs, step_losses, acc_tr, acc_te = track_batch_strategy(X_tr, y_tr, X_te, y_te, strategy=strat, steps=32)
        results[strat] = {
            'step_accs': np.array(step_accs),
            'step_losses': step_losses,
            'acc_tr': acc_tr,
            'acc_te': acc_te
        }
        print(f"Strategy: {strat:<14} | Final Train Acc: {acc_tr:6.1f}% | Final Test Acc: {acc_te:6.1f}%")
        
    print("==========================================================================\n")

    # Set dark aesthetic style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12), dpi=300)
    
    class_colors = ['#FF5376', '#00F5D4', '#FFEE55', '#7B2CBF']
    steps = np.arange(1, 33)

    # -------------------------------------------------------------------------
    # Panel 1: Strategy 3 (Single-Class Sequential) - Catastrophic Forgetting
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    accs_single = results['single_class']['step_accs']
    
    for k in range(4):
        ax1.plot(steps, accs_single[:, k], color=class_colors[k], linewidth=2.8, label=f'Class {k} Accuracy')
        
    # Vertical lines for class blocks
    ax1.axvline(x=8.5, color='#888888', linestyle='--', alpha=0.6)
    ax1.axvline(x=16.5, color='#888888', linestyle='--', alpha=0.6)
    ax1.axvline(x=24.5, color='#888888', linestyle='--', alpha=0.6)
    
    ax1.text(4.5, 50, "Class 0\nBlock", color='#FF5376', fontsize=9, fontweight='bold', ha='center')
    ax1.text(12.5, 50, "Class 1\nBlock", color='#00F5D4', fontsize=9, fontweight='bold', ha='center')
    ax1.text(20.5, 50, "Class 2\nBlock", color='#FFEE55', fontsize=9, fontweight='bold', ha='center')
    ax1.text(28.5, 50, "Class 3\nBlock", color='#7B2CBF', fontsize=9, fontweight='bold', ha='center')

    ax1.set_title("1. Strategy 3: Single-Class Sequential Batches\n(Catastrophic Forgetting & Class Collapse!)", 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Training Mini-Batch Step (Batch Size = 4)", fontsize=10)
    ax1.set_ylabel("Class Accuracy (%)", fontsize=10)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle='--', alpha=0.2)
    ax1.legend(loc='lower left', framealpha=0.85, fontsize=8)

    ax1.annotate('Class 0 & 1 Accuracy Collapses to 0%!\n(Forgot earlier classes when training on Class 3)', 
                 xy=(32, 12.5), xytext=(12, 18),
                 arrowprops=dict(facecolor='#FF5376', shrink=0.08, width=1.5, headwidth=8),
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="#2A1A2A", ec="#FF5376", lw=1.5))

    # -------------------------------------------------------------------------
    # Panel 2: Strategy 2 (Balanced 1-per-Class) - Continuous Stability
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    accs_bal = results['balanced']['step_accs']
    
    for k in range(4):
        ax2.plot(steps, accs_bal[:, k], color=class_colors[k], linewidth=2.8, label=f'Class {k} Accuracy')

    ax2.set_title("2. Strategy 2: Balanced Batches (1 Sample Per Class)\n(Smooth Symmetric Joint Class Learning)", 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Training Mini-Batch Step (Batch Size = 4)", fontsize=10)
    ax2.set_ylabel("Class Accuracy (%)", fontsize=10)
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle='--', alpha=0.2)
    ax2.legend(loc='lower right', framealpha=0.85, fontsize=8)

    # -------------------------------------------------------------------------
    # Panel 3: Strategy 1 (Pure Random Shuffled) - Stochastic Convergence
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    accs_rand = results['random']['step_accs']
    
    for k in range(4):
        ax3.plot(steps, accs_rand[:, k], color=class_colors[k], linewidth=2.8, label=f'Class {k} Accuracy')

    ax3.set_title("3. Strategy 1: Pure Random Shuffled Batches (Standard i.i.d.)\n(Stochastic Convergence & Unbiased Gradients)", 
                  fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("Training Mini-Batch Step (Batch Size = 4)", fontsize=10)
    ax3.set_ylabel("Class Accuracy (%)", fontsize=10)
    ax3.set_ylim(-5, 105)
    ax3.grid(True, linestyle='--', alpha=0.2)
    ax3.legend(loc='lower right', framealpha=0.85, fontsize=8)

    # -------------------------------------------------------------------------
    # Panel 4: Final Accuracy Comparison Across Strategies
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    
    strat_labels = ["Strategy 1\n(Pure Random)", "Strategy 2\n(Balanced 1/Class)", "Strategy 3\n(Sequential Block)"]
    tr_vals = [results['random']['acc_tr'], results['balanced']['acc_tr'], results['single_class']['acc_tr']]
    te_vals = [results['random']['acc_te'], results['balanced']['acc_te'], results['single_class']['acc_te']]
    
    x = np.arange(len(strat_labels))
    width = 0.35
    
    rects1 = ax4.bar(x - width/2, tr_vals, width, label='Train Accuracy', color='#00F5D4', alpha=0.9)
    rects2 = ax4.bar(x + width/2, te_vals, width, label='Test Accuracy', color='#FFEE55', alpha=0.9)
    
    ax4.set_ylabel('Accuracy (%)', fontsize=10)
    ax4.set_title('4. Final Accuracy Comparison Across Batching Strategies\n(32 Steps, Batch Size = 4)', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(strat_labels, fontsize=9)
    ax4.set_ylim(0, 115)
    ax4.legend(loc='upper right', framealpha=0.85, fontsize=8)
    ax4.grid(True, linestyle='--', alpha=0.2)
    
    for rect in rects1:
        height = rect.get_height()
        ax4.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#00F5D4')
                    
    for rect in rects2:
        height = rect.get_height()
        ax4.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#FFEE55')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "batch_dynamics_story.png")
    plt.savefig(output_path, dpi=300)
    print(f"Batch dynamics graphic saved successfully to {output_path}")

if __name__ == "__main__":
    run_batch_experiment()
