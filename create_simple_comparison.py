"""
Simple Model Comparison - IndoBERT vs Traditional ML

Creates a focused comparison visualization showing the key results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = Path("comprehensive_results")
INDOBERT_RESULTS = Path("indobert_experiments_results.csv")
OUTPUT_DIR = Path("comprehensive_results/final_comparison")
OUTPUT_DIR.mkdir(exist_ok=True)

def create_simple_comparison():
    """Create a simple, focused comparison chart."""
    
    # Data from experiments
    models = ['SVM', 'Random Forest', 'Naive Bayes', 'Logistic Regression', 'IndoBERT']
    f1_scores = [98.18, 97.52, 94.51, 93.27, 99.40]  # Best F1 scores from each model
    training_times = [11.4, 273.3, 0.17, 2.6, 16693.3]  # Training times in seconds
    
    # Create figure with reasonable size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Model Performance Comparison: Traditional ML vs IndoBERT', fontsize=14, fontweight='bold')
    
    # Colors - red for IndoBERT, blue for traditional models
    colors = ['#3498db', '#3498db', '#3498db', '#3498db', '#e74c3c']
    
    # 1. F1 Score Comparison
    bars1 = ax1.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('F1-Score Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1-Score (%)', fontsize=11)
    ax1.set_ylim(90, 100)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars1, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{score:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Highlight best model
    best_idx = np.argmax(f1_scores)
    bars1[best_idx].set_edgecolor('gold')
    bars1[best_idx].set_linewidth(3)
    
    # 2. Training Time Comparison (Log Scale)
    bars2 = ax2.bar(models, training_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Training Time Comparison (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)', fontsize=11)
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, time in zip(bars2, training_times):
        height = bar.get_height()
        if time > 3600:
            time_str = f'{time/3600:.1f}h'
        elif time > 60:
            time_str = f'{time/60:.1f}m'
        else:
            time_str = f'{time:.1f}s'
        
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.5,
                time_str, ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'simple_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Simple comparison saved to: {OUTPUT_DIR / 'simple_model_comparison.png'}")
    
    # Create a summary table
    create_summary_table(models, f1_scores, training_times)

def create_summary_table(models, f1_scores, training_times):
    """Create a summary table as text."""
    
    summary_text = "# Model Performance Summary\n\n"
    summary_text += "| Model | F1-Score | Training Time |\n"
    summary_text += "|-------|----------|---------------|\n"
    
    for model, f1, time in zip(models, f1_scores, training_times):
        if time > 3600:
            time_str = f"{time/3600:.1f} hours"
        elif time > 60:
            time_str = f"{time/60:.1f} minutes"
        else:
            time_str = f"{time:.1f} seconds"
        
        summary_text += f"| {model} | {f1:.1f}% | {time_str} |\n"
    
    summary_text += "\n## Key Findings\n\n"
    summary_text += f"1. **Best Performance**: {models[np.argmax(f1_scores)]} with {max(f1_scores):.1f}% F1-score\n"
    summary_text += f"2. **Fastest Training**: {models[np.argmin(training_times)]} with {min(training_times):.1f} seconds\n"
    summary_text += f"3. **IndoBERT Advantage**: {f1_scores[-1] - max(f1_scores[:-1]):.2f}% higher F1 than best traditional model\n"
    summary_text += f"4. **Speed vs Accuracy Trade-off**: IndoBERT takes {training_times[-1]/min(training_times):.0f}x longer but achieves superior accuracy\n"
    
    with open(OUTPUT_DIR / 'model_summary.md', 'w') as f:
        f.write(summary_text)
    
    print(f"✅ Summary table saved to: {OUTPUT_DIR / 'model_summary.md'}")

def main():
    """Main function."""
    
    print("🚀 Creating simple model comparison...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Create comparison
    create_simple_comparison()
    
    print("\n📊 Comparison Results:")
    print("=" * 50)
    print("IndoBERT achieved 99.40% F1-score")
    print("Best traditional model (SVM): 98.18% F1-score")
    print("IndoBERT advantage: +1.22% F1-score")
    print("Training time trade-off: 4.6 hours vs 11.4 seconds")
    print("=" * 50)
    
    print(f"\n✅ All outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()