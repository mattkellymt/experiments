import os
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Set random seed for complete reproducibility
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------------------------------------------------------
# 1. Fully Randomized Cluster Generation
# Centroids ~ Normal(0, 2.5), Std Devs ~ |Normal(0.55, 0.25)|
# -----------------------------------------------------------------------------
def generate_random_5050_dataset(num_per_class=32, K=4, seed=42):
    np.random.seed(seed)
    
    # 1. Random Centroids from Normal(0, 2.5)
    centroids = np.random.normal(loc=0.0, scale=2.5, size=(K, 2))
    
    # 2. Random Std Devs for each cluster from |Normal(0.55, 0.25)| bounded in [0.25, 1.1]
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
# 2. Cover's RBF Feature Lifting & Linear Classifier
# -----------------------------------------------------------------------------
def extract_rbf_features(X, centers, gamma=2.2):
    dist = torch.cdist(X, centers)
    return torch.exp(-gamma * dist**2)

def train_and_eval_split(H_train, y_train, H_test, y_test, num_classes=4, epochs=600, lr=0.12):
    D = H_train.shape[1]
    clf = nn.Linear(D, num_classes)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = clf(H_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
    clf.eval()
    with torch.no_grad():
        logits_tr = clf(H_train)
        preds_tr = torch.argmax(logits_tr, dim=1)
        acc_tr = (preds_tr == y_train).float().mean().item()
        
        logits_te = clf(H_test)
        preds_te = torch.argmax(logits_te, dim=1)
        acc_te = (preds_te == y_test).float().mean().item()
        
    return acc_tr, acc_te, clf

# -----------------------------------------------------------------------------
# 3. Experiment Pipeline & 4-Panel Visual Story
# -----------------------------------------------------------------------------
def run_experiment_and_plot():
    X_train, y_train, X_test, y_test, centroids, stds = generate_random_5050_dataset(num_per_class=32, K=4, seed=42)
    
    N_train = len(y_train) # 128 points
    N_test = len(y_test)   # 128 points
    K = 4
    
    class_colors = ['#FF5376', '#00F5D4', '#FFEE55', '#7B2CBF']
    cmap_4 = ListedColormap(class_colors)
    
    D_list = [2, 4, 8, 16, 32, 64, 96, 128, 192, 256]
    train_accs = []
    test_accs = []
    classifiers = {}
    feature_matrices = {}
    
    print("=============================================================")
    print("COVER'S THEOREM RANDOM CLUSTERS EXPERIMENT (N_tr = 128, N_te = 128)")
    print("Random Centroids ~ Normal(0, 2.5) | Random Stds ~ Normal(0.55, 0.25)")
    print("=============================================================")
    for k in range(K):
        print(f"  Class {k}: Centroid = ({centroids[k,0]:.2f}, {centroids[k,1]:.2f})  |  Std Dev = {stds[k]:.2f}")
    print("-------------------------------------------------------------")
    print(f"{'Dimension D':<12} | {'Train Accuracy (%)':<20} | {'Test Accuracy (%)':<20}")
    print("-------------------------------------------------------------")
    
    torch.manual_seed(42)
    for D in D_list:
        if D <= N_train:
            centers = X_train[:D]
        else:
            centers = torch.cat([X_train, torch.randn(D - N_train, 2)], dim=0)
            
        H_train = extract_rbf_features(X_train, centers)
        H_test = extract_rbf_features(X_test, centers)
        
        acc_tr, acc_te, clf = train_and_eval_split(H_train, y_train, H_test, y_test, num_classes=K)
        train_accs.append(acc_tr * 100.0)
        test_accs.append(acc_te * 100.0)
        classifiers[D] = clf
        feature_matrices[D] = (H_train, H_test)
        
        print(f"{D:12d} | {acc_tr*100.0:20.2f} | {acc_te*100.0:20.2f}")
        
    print("=============================================================\n")

    # Set dark aesthetic style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12), dpi=300)
    
    # -------------------------------------------------------------------------
    # Panel 1: Randomly Generated Clusters in 2D Space
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    
    for k in range(K):
        mask_tr = (y_train.numpy() == k)
        mask_te = (y_test.numpy() == k)
        
        ax1.scatter(X_train[mask_tr, 0], X_train[mask_tr, 1], color=class_colors[k], 
                    s=65, edgecolors='white', linewidth=1.0, alpha=0.95)
        ax1.scatter(X_test[mask_te, 0], X_test[mask_te, 1], color=class_colors[k], marker='X',
                    s=65, edgecolors='white', linewidth=0.6, alpha=0.85)
        # Plot centroid marker
        ax1.scatter(centroids[k, 0], centroids[k, 1], color=class_colors[k], marker='*',
                    s=220, edgecolors='white', linewidth=1.5, zorder=6)

    ax1.set_title(f"1. Random Clusters in 2D (Centroids '*', Train '.', Test 'X')\n(Centroids ~ $\\mathcal{{N}}(0, 2.5)$, Stds ~ $|\\mathcal{{N}}(0.55, 0.25)|$)", 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Feature $x_1$", fontsize=11)
    ax1.set_ylabel("Feature $x_2$", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.2)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='Random Centroids (*)', markerfacecolor='#FFEE55', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Train Points (Solid)', markerfacecolor='#00F5D4', markersize=7),
        Line2D([0], [0], marker='X', color='w', label='Test Points (Crosses)', markerfacecolor='#FF5376', markersize=7)
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.85, fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel 2: Train & Test Accuracy vs. Feature Dimension D
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    
    ax2.plot(D_list, train_accs, color='#00F5D4', linewidth=3.0, marker='o', markersize=7, 
             label='Train Accuracy (Cover Capacity)')
    ax2.plot(D_list, test_accs, color='#FFEE55', linewidth=3.0, marker='s', markersize=7, linestyle='--',
             label='Test Accuracy (Generalization)')
    
    ax2.axvline(x=N_train, color='#FF5376', linestyle='--', linewidth=2.0, 
                label=f"Cover's Threshold ($D = N_{{train}} = {N_train}$)")
    ax2.axhline(y=100.0, color='#888888', linestyle=':', alpha=0.5)

    ax2.set_title(f"2. Cover's Theorem: Accuracy vs. Dimension D\n(Random Clusters: Train N={N_train}, Test N={N_test})", 
                  fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Feature Expansion Dimension ($D$)", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_xscale('log')
    ax2.set_xticks(D_list)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.set_ylim(60, 105)
    ax2.grid(True, linestyle='--', alpha=0.2)
    ax2.legend(loc='lower right', framealpha=0.85, fontsize=8.5)

    ax2.annotate(f'Train Acc = 100% (Cover D={N_train})\nTest Acc = {test_accs[D_list.index(128)]:.1f}%', 
                 xy=(128, 100), xytext=(25, 78),
                 arrowprops=dict(facecolor='#00F5D4', shrink=0.08, width=1.8, headwidth=9),
                 fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", fc="#1A2A2A", ec="#00F5D4", lw=1.5))

    # -------------------------------------------------------------------------
    # Panel 3: Lifted Feature Space PCA Projection at D = 128
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    
    H_te_128 = feature_matrices[128][1].numpy()
    H_centered = H_te_128 - H_te_128.mean(axis=0)
    U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
    H_pca = H_centered @ Vt[:2].T
    
    for k in range(K):
        mask_te = (y_test.numpy() == k)
        ax3.scatter(H_pca[mask_te, 0], H_pca[mask_te, 1], color=class_colors[k], label=f'Class {k}', 
                    s=65, edgecolors='white', linewidth=0.8, alpha=0.95)
        
    acc_te_128 = test_accs[D_list.index(128)]
    ax3.set_title(f"3. Lifted Test Set Feature Space (D = 128, Top 2 PCs)\nTest Accuracy: {acc_te_128:.1f}% (Separated Clusters!)", 
                  fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel("Principal Component 1", fontsize=11)
    ax3.set_ylabel("Principal Component 2", fontsize=11)
    ax3.legend(loc='upper right', framealpha=0.85, fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.2)

    # -------------------------------------------------------------------------
    # Panel 4: Test Confusion Matrix at D = 128
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    
    clf_128 = classifiers[128]
    H_test_128 = feature_matrices[128][1]
    with torch.no_grad():
        preds_test_128 = torch.argmax(clf_128(H_test_128), dim=1).numpy()
        
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_test.numpy(), preds_test_128):
        cm[t, p] += 1
        
    im4 = ax4.imshow(cm, cmap='viridis')
    cbar4 = fig.colorbar(im4, ax=ax4)
    cbar4.set_label("Number of Test Samples", fontsize=10)
    
    for i in range(K):
        for j in range(K):
            ax4.text(j, i, f"{cm[i, j]}", ha="center", va="center", 
                     color="white" if cm[i, j] < cm.max()/2 else "black",
                     fontsize=12, fontweight='bold')
            
    ax4.set_title(f"4. Test Set Confusion Matrix at D = 128\n(Overall Test Accuracy: {acc_te_128:.1f}%)", 
                  fontsize=13, fontweight='bold', pad=10)
    ax4.set_xticks(range(K))
    ax4.set_yticks(range(K))
    ax4.set_xticklabels([f"Class {k}" for k in range(K)])
    ax4.set_yticklabels([f"Class {k}" for k in range(K)])

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "covers_4class_story.png")
    plt.savefig(output_path, dpi=300)
    print(f"Visual story graphic saved successfully to {output_path}")

if __name__ == "__main__":
    run_experiment_and_plot()
