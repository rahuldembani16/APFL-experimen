import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

LOG_DIR = "./log"
ALGORITHMS = ["JADE-FL-(our)", "FLAvg", "Krum", "Median", "TrimMean"]
ATTACKS = ["NA", "LFA", "MPA"]
DATASET = "plantvillage"
POISONING_RATIOS = {"NA": "0", "LFA": "0.3", "MPA": "0.3"}
N_COMM = 50

# Enhanced styling configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['grid.color'] = '#e9ecef'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 18


def load_experiment_data(algorithm, attack):
    pr = POISONING_RATIOS[attack]
    filename = f"{algorithm}_{DATASET}_{attack}_pr{pr}_E1_ncomm{N_COMM}.csv"
    filepath = os.path.join(LOG_DIR, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print(f"File not found: {filename}")
        return None


def plot_accuracy_comparison(attack):
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Color palette for algorithms
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C']
    markers = ['o', 's', '^', 'D', 'v']
    
    min_acc, max_acc = 1, 0
    
    for i, algorithm in enumerate(ALGORITHMS):
        data = load_experiment_data(algorithm, attack)
        if data is not None:
            acc = data["accuracy"]
            min_acc = min(min_acc, acc.min())
            max_acc = max(max_acc, acc.max())
            
            # Smooth the data using rolling average
            rounds = [int(r.replace("round[","").replace("]", "")) for r in data["round"]]
            
            # Plot with enhanced styling
            ax.plot(rounds, acc, 
                   color=colors[i], 
                   marker=markers[i], 
                   linewidth=3, 
                   markersize=6, 
                   label=algorithm,
                   markerfacecolor=colors[i],
                   markeredgecolor='white',
                   markeredgewidth=2,
                   alpha=0.9)
    
    # Enhanced title and labels
    attack_display = {"NA": "No Attack", "LFA": "Label Flipping Attack", "MPA": "Model Poisoning Attack"}
    ax.set_title(f'Accuracy Comparison - {attack_display[attack]}', 
                fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Communication Round', fontsize=14, fontweight='semibold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='semibold')
    
    # Enhanced legend
    legend = ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, 
                      loc='best', borderpad=1, labelspacing=0.5)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Set limits with better margins
    ax.set_xlim(0, N_COMM)
    ax.set_ylim(max(0, min_acc - 0.02), min(1, max_acc + 0.02))
    
    # Add subtle background
    ax.set_facecolor('#fafbfc')
    
    # Enhanced spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#d1d5db')
    
    plt.tight_layout()
    plt.savefig(f"accuracy_comparison_{attack}.png", dpi=300, bbox_inches="tight", 
                facecolor='white', edgecolor='none')
    plt.close()


def plot_loss_comparison(attack):
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Color palette for algorithms (same as accuracy for consistency)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C']
    markers = ['s', 'o', '^', 'D', 'v']  # Different markers for loss
    
    min_loss, max_loss = float('inf'), float('-inf')
    
    for i, algorithm in enumerate(ALGORITHMS):
        data = load_experiment_data(algorithm, attack)
        if data is not None:
            loss = data["loss"]
            min_loss = min(min_loss, loss.min())
            max_loss = max(max_loss, loss.max())
            
            rounds = [int(r.replace("round[","").replace("]", "")) for r in data["round"]]
            
            # Plot with enhanced styling
            ax.plot(rounds, loss, 
                   color=colors[i], 
                   marker=markers[i], 
                   linewidth=3, 
                   markersize=6, 
                   label=algorithm,
                   markerfacecolor=colors[i],
                   markeredgecolor='white',
                   markeredgewidth=2,
                   alpha=0.9,
                   linestyle='--')  # Dashed line for loss
    
    # Enhanced title and labels
    attack_display = {"NA": "No Attack", "LFA": "Label Flipping Attack", "MPA": "Model Poisoning Attack"}
    ax.set_title(f'Loss Comparison - {attack_display[attack]}', 
                fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Communication Round', fontsize=14, fontweight='semibold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='semibold')
    
    # Enhanced legend
    legend = ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, 
                      loc='best', borderpad=1, labelspacing=0.5)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Set limits with better margins
    ax.set_xlim(0, N_COMM)
    ax.set_ylim(max(0, min_loss - 0.1), max_loss + 0.1)
    
    # Add subtle background
    ax.set_facecolor('#fafbfc')
    
    # Enhanced spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#d1d5db')
    
    plt.tight_layout()
    plt.savefig(f"loss_comparison_{attack}.png", dpi=300, bbox_inches="tight", 
                facecolor='white', edgecolor='none')
    plt.close()


def plot_final_accuracy_bar():
    final_accuracies = {attack: [] for attack in ATTACKS}
    for attack in ATTACKS:
        for algorithm in ALGORITHMS:
            data = load_experiment_data(algorithm, attack)
            if data is not None:
                final_accuracies[attack].append(data["accuracy"].iloc[-1])
            else:
                final_accuracies[attack].append(0)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Modern color palette for attacks
    attack_colors = ['#4CAF50', '#FF9800', '#F44336']  # Green, Orange, Red
    width = 0.25
    x = np.arange(len(ALGORITHMS))
    
    # Create grouped bar chart with enhanced styling
    bars = []
    for i, attack in enumerate(ATTACKS):
        bar = ax.bar(x + i*width, final_accuracies[attack], width, 
                    label=f"{attack} Attack", 
                    color=attack_colors[i], 
                    alpha=0.85,
                    edgecolor='white',
                    linewidth=2,
                    capsize=5)
        bars.append(bar)
    
    # Add value labels on top of bars
    for i, attack in enumerate(ATTACKS):
        for j, value in enumerate(final_accuracies[attack]):
            if value > 0:  # Only show non-zero values
                ax.text(j + i*width, value + 0.01, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Enhanced title and labels
    ax.set_title('Final Accuracy Comparison Across Algorithms and Attacks', 
                fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel('Algorithms', fontsize=14, fontweight='semibold')
    ax.set_ylabel('Final Accuracy', fontsize=14, fontweight='semibold')
    
    # Enhanced x-axis
    ax.set_xticks(x + width)
    ax.set_xticklabels(ALGORITHMS, fontsize=12, rotation=0)
    
    # Enhanced legend
    legend = ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, 
                      loc='best', borderpad=1, labelspacing=0.5)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.1)
    
    # Add subtle background
    ax.set_facecolor('#fafbfc')
    
    # Enhanced spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#d1d5db')
    
    # Add attack description text
    attack_descriptions = {
        "NA": "No Attack",
        "LFA": "Label Flipping Attack", 
        "MPA": "Model Poisoning Attack"
    }
    
    # Add a subtle text box with attack descriptions
    textstr = '\n'.join([f'{attack}: {desc}' for attack, desc in attack_descriptions.items()])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig("final_accuracy_comparison.png", dpi=300, bbox_inches="tight", 
                facecolor='white', edgecolor='none')
    plt.close()


def create_comprehensive_dashboard():
    """Create a comprehensive dashboard with all metrics in one view"""
    fig = plt.figure(figsize=(20, 16))
    
    # Create a 3x3 grid for different visualizations
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Colors for algorithms
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C']
    
    # 1. Accuracy trends for all attacks (top row)
    for attack_idx, attack in enumerate(ATTACKS):
        ax = fig.add_subplot(gs[0, attack_idx])
        
        for i, algorithm in enumerate(ALGORITHMS):
            data = load_experiment_data(algorithm, attack)
            if data is not None:
                acc = data["accuracy"]
                rounds = [int(r.replace("round[","").replace("]", "")) for r in data["round"]]
                ax.plot(rounds, acc, color=colors[i], linewidth=2, label=algorithm)
        
        attack_display = {"NA": "No Attack", "LFA": "Label Flipping", "MPA": "Model Poisoning"}
        ax.set_title(f'Accuracy - {attack_display[attack]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
    
    # 2. Loss trends for all attacks (middle row)
    for attack_idx, attack in enumerate(ATTACKS):
        ax = fig.add_subplot(gs[1, attack_idx])
        
        for i, algorithm in enumerate(ALGORITHMS):
            data = load_experiment_data(algorithm, attack)
            if data is not None:
                loss = data["loss"]
                rounds = [int(r.replace("round[","").replace("]", "")) for r in data["round"]]
                ax.plot(rounds, loss, color=colors[i], linewidth=2, label=algorithm, linestyle='--')
        
        ax.set_title(f'Loss - {attack_display[attack]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # 3. Algorithm performance comparison (bottom row, spans 2 columns)
    ax = fig.add_subplot(gs[2, :2])
    
    # Calculate average performance across all attacks for each algorithm
    avg_accuracies = []
    std_accuracies = []
    
    for algorithm in ALGORITHMS:
        accuracies = []
        for attack in ATTACKS:
            data = load_experiment_data(algorithm, attack)
            if data is not None:
                accuracies.append(data["accuracy"].iloc[-1])
        if accuracies:
            avg_accuracies.append(np.mean(accuracies))
            std_accuracies.append(np.std(accuracies))
        else:
            avg_accuracies.append(0)
            std_accuracies.append(0)
    
    bars = ax.bar(ALGORITHMS, avg_accuracies, yerr=std_accuracies, 
                  color=colors, alpha=0.8, capsize=5, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for i, (bar, avg, std) in enumerate(zip(bars, avg_accuracies, std_accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{avg:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Average Final Accuracy Across All Attacks', fontsize=16, fontweight='bold')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Average Final Accuracy')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # 4. Summary statistics (bottom right)
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    # Create summary text
    summary_text = "EXPERIMENT SUMMARY\n" + "="*30 + "\n\n"
    summary_text += f"Dataset: {DATASET}\n"
    summary_text += f"Communication Rounds: {N_COMM}\n"
    summary_text += f"Algorithms Tested: {len(ALGORITHMS)}\n"
    summary_text += f"Attack Types: {len(ATTACKS)}\n\n"
    
    # Find best performing algorithm
    if avg_accuracies:
        best_idx = np.argmax(avg_accuracies)
        best_algo = ALGORITHMS[best_idx]
        best_acc = avg_accuracies[best_idx]
        summary_text += f"Best Algorithm: {best_algo}\n"
        summary_text += f"Best Avg Accuracy: {best_acc:.3f}\n\n"
    
    summary_text += "Generated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Main title
    fig.suptitle('Federated Learning Performance Dashboard', fontsize=24, fontweight='bold', y=0.98)
    
    plt.savefig("comprehensive_dashboard.png", dpi=300, bbox_inches="tight", 
                facecolor='white', edgecolor='none')
    plt.close()


def main():
    for attack in ATTACKS:
        plot_accuracy_comparison(attack)
        plot_loss_comparison(attack)
    plot_final_accuracy_bar()
    create_comprehensive_dashboard()
    print("All comparison graphs generated.")
    print("Enhanced visualizations created with:")
    print("- Improved color schemes and styling")
    print("- Better legends and annotations")
    print("- Comprehensive dashboard view")
    print("- Value labels on charts")
    print("- Professional presentation quality")


if __name__ == "__main__":
    main()