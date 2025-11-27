"""plotting utilities for evaluation results"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_comparison_chart(
    results: Dict[str, Dict[str, float]],
    save_path: str = "results/comparison_chart.png"
):
    """grouped bar chart comparing all systems across metrics"""
    systems = list(results.keys())
    metrics = ['precision', 'recall', 'f1_score', 'accuracy']
    metric_labels = ['Precision', 'Recall', 'F1 Score', 'Accuracy']

    data = {metric: [results[sys][metric] for sys in systems] for metric in metrics}

    x = np.arange(len(systems))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, data[metric], width, label=label, color=colors[i], alpha=0.8)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('System', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Code Plagiarism Detection System Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=12)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"saved comparison chart to: {save_path}")
    plt.close()


def plot_ablation_study(
    k_values: List[int],
    results: Dict[str, List[float]],
    save_path: str = "results/ablation_k_values.png",
    title: str = "Ablation Study: Impact of k on RAG Performance"
):
    """line plot showing how metrics change with different k values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'precision': '#2ecc71', 'recall': '#3498db', 'f1_score': '#e74c3c', 'accuracy': '#f39c12'}

    for metric, scores in results.items():
        if metric in ['precision', 'recall', 'f1_score', 'accuracy']:
            ax.plot(k_values, scores, marker='o', linewidth=2,
                   label=metric.replace('_', ' ').title(),
                   color=colors.get(metric, '#95a5a6'))

    ax.set_xlabel('k (Number of Retrieved Documents)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"saved ablation plot to: {save_path}")
    plt.close()


def plot_confusion_matrices(
    results: Dict[str, Dict[str, int]],
    save_path: str = "results/confusion_matrices.png"
):
    """heatmaps showing tp/fp/fn/tn for each system side by side"""
    n_systems = len(results)
    fig, axes = plt.subplots(1, n_systems, figsize=(5*n_systems, 4))

    if n_systems == 1:
        axes = [axes]

    for ax, (system_name, metrics) in zip(axes, results.items()):
        cm = np.array([
            [metrics['true_positives'], metrics['false_negatives']],
            [metrics['false_positives'], metrics['true_negatives']]
        ])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Predicted Pos', 'Predicted Neg'],
                   yticklabels=['Actual Pos', 'Actual Neg'],
                   cbar=False)

        ax.set_title(f'{system_name}\n(Accuracy: {metrics.get("accuracy", 0):.3f})',
                    fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"saved confusion matrices to: {save_path}")
    plt.close()


def plot_cost_vs_performance(
    systems: List[str],
    f1_scores: List[float],
    costs: List[float],
    save_path: str = "results/cost_vs_performance.png"
):
    """scatter plot showing tradeoff between cost and f1 score"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

    for i, (system, f1, cost) in enumerate(zip(systems, f1_scores, costs)):
        ax.scatter(cost, f1, s=300, alpha=0.7, color=colors[i % len(colors)],
                  edgecolors='black', linewidth=2)
        ax.annotate(system, (cost, f1), fontsize=11, fontweight='bold',
                   xytext=(10, 10), textcoords='offset points')

    ax.set_xlabel('Relative Cost', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Cost vs Performance Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"saved cost vs performance plot to: {save_path}")
    plt.close()
