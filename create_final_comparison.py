"""
Create Final Comparison Visualization - All Models Including IndoBERT

This script creates comprehensive visualizations comparing all models:
- Traditional ML models (SVM, Random Forest, Naive Bayes, Logistic Regression)
- IndoBERT transformer model
- Performance metrics, training time, and model characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configuration
RESULTS_DIR = Path("comprehensive_results")
INDOBERT_RESULTS = Path("indobert_experiments_results.csv")
INDOBERT_SUMMARY = Path("indobert_experiments/experiment_summary.json")
OUTPUT_DIR = Path("comprehensive_results/final_comparison")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_comprehensive_results():
    """Load the comprehensive experiment results."""
    try:
        df = pd.read_csv(RESULTS_DIR / "comprehensive_experiment_summary_merged.csv")
        return df
    except FileNotFoundError:
        print("Comprehensive results CSV not found. Creating synthetic data for demonstration.")
        # Create synthetic data for demonstration
        data = {
            'model': ['svm', 'svm', 'rf', 'rf', 'nb', 'nb', 'logreg', 'logreg'],
            'param_name': ['c_value', 'c_value', 'n_estimators', 'n_estimators', 'alpha', 'alpha', 'c_value', 'c_value'],
            'param_value': [1.0, 0.1, 500, 200, 0.1, 1.0, 1.0, 10.0],
            'max_features': [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
            'accuracy_mean': [0.9830, 0.9802, 0.9769, 0.9767, 0.9496, 0.9468, 0.9331, 0.9327],
            'f1_mean': [0.9818, 0.9787, 0.9752, 0.9750, 0.9451, 0.9424, 0.9327, 0.9323],
            'training_duration': [11.4, 6.7, 273.3, 132.9, 0.17, 0.18, 2.6, 2.4]
        }
        return pd.DataFrame(data)

def load_indobert_results():
    """Load IndoBERT experiment results."""
    try:
        df = pd.read_csv(INDOBERT_RESULTS)
        # Filter successful experiments
        successful = df[df['success'] == True].copy()
        
        if len(successful) > 0:
            # Add model type and clean up
            successful['model'] = 'indobert'
            successful['param_name'] = 'learning_rate'
            successful['param_value'] = successful['learning_rate']
            successful['max_features'] = successful.get('max_length', 128)
            
            # Select relevant columns and rename to match comprehensive format
            result = successful[['model', 'param_name', 'param_value', 'max_features', 
                               'accuracy', 'f1', 'training_duration']].copy()
            result = result.rename(columns={
                'accuracy': 'accuracy_mean',
                'f1': 'f1_mean'
            })
            return result
        else:
            print("No successful IndoBERT experiments found.")
            return pd.DataFrame()
            
    except FileNotFoundError:
        print("IndoBERT results not found. Using synthetic data for demonstration.")
        # Synthetic IndoBERT data for demonstration
        data = {
            'model': ['indobert', 'indobert'],
            'param_name': ['learning_rate', 'learning_rate'],
            'param_value': [1e-05, 2e-05],
            'max_features': [128, 128],
            'accuracy_mean': [0.9940, 0.9936],
            'f1_mean': [0.9940, 0.9936],
            'training_duration': [16693.3, 14285.4]
        }
        return pd.DataFrame(data)

def create_comprehensive_comparison():
    """Create comprehensive comparison visualizations."""
    
    # Load data
    traditional_df = load_comprehensive_results()
    indobert_df = load_indobert_results()
    
    # Combine datasets
    if len(indobert_df) > 0:
        # Get best results from each model type
        best_traditional = traditional_df.loc[traditional_df.groupby('model')['f1_mean'].idxmax()]
        best_indobert = indobert_df.loc[indobert_df.groupby('model')['f1_mean'].idxmax()]
        
        # Combine best results
        comparison_df = pd.concat([best_traditional, best_indobert], ignore_index=True)
    else:
        best_traditional = traditional_df.loc[traditional_df.groupby('model')['f1_mean'].idxmax()]
        comparison_df = best_traditional
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Model Comparison: Traditional ML vs IndoBERT', fontsize=16, fontweight='bold')
    
    # 1. Performance Comparison (F1 Score)
    ax1 = axes[0, 0]
    model_names = []
    f1_scores = []
    colors = []
    
    for _, row in comparison_df.iterrows():
        if row['model'] == 'indobert':
            model_names.append('IndoBERT')
            colors.append('#FF6B6B')  # Red for IndoBERT
        else:
            model_names.append(row['model'].upper())
            colors.append('#4ECDC4')  # Teal for traditional models
        
        f1_scores.append(row['f1_mean'] * 100)  # Convert to percentage
    
    bars = ax1.bar(model_names, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1-Score (%)', fontsize=12)
    ax1.set_ylim(90, 100)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Highlight best model
    best_idx = np.argmax(f1_scores)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # 2. Training Time Comparison (Log Scale)
    ax2 = axes[0, 1]
    training_times = comparison_df['training_duration'].values
    model_names_time = []
    
    for model in comparison_df['model']:
        if model == 'indobert':
            model_names_time.append('IndoBERT')
        else:
            model_names_time.append(model.upper())
    
    bars2 = ax2.bar(model_names_time, training_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Training Time Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.set_yscale('log')
    
    # Add value labels
    for bar, time in zip(bars2, training_times):
        height = bar.get_height()
        if time > 1000:
            time_str = f'{time/3600:.1f}h'
        elif time > 60:
            time_str = f'{time/60:.1f}m'
        else:
            time_str = f'{time:.1f}s'
        
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                time_str, ha='center', va='bottom', fontweight='bold')
    
    # 3. Accuracy vs Training Time Scatter Plot
    ax3 = axes[1, 0]
    x = comparison_df['training_duration'].values
    y = comparison_df['accuracy_mean'].values * 100
    
    scatter = ax3.scatter(x, y, c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)
    ax3.set_title('Accuracy vs Training Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Training Time (seconds)', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_xscale('log')
    ax3.set_ylim(90, 100)
    
    # Add model labels
    for i, (xi, yi, model) in enumerate(zip(x, y, model_names_time)):
        ax3.annotate(model, (xi, yi), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    # 4. Model Performance Summary Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    summary_data = []
    for _, row in comparison_df.iterrows():
        model_name = 'IndoBERT' if row['model'] == 'indobert' else row['model'].upper()
        time_str = f"{row['training_duration']/3600:.1f}h" if row['training_duration'] > 3600 else f"{row['training_duration']/60:.1f}m"
        
        summary_data.append([
            model_name,
            f"{row['accuracy_mean']*100:.2f}%",
            f"{row['f1_mean']*100:.2f}%",
            time_str
        ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Model', 'Accuracy', 'F1-Score', 'Training Time'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#E8E8E8')
                cell.set_text_props(weight='bold')
            else:
                if summary_data[i-1][0] == 'IndoBERT':
                    cell.set_facecolor('#FFE5E5')  # Light red for IndoBERT
                else:
                    cell.set_facecolor('#E5F5F5')  # Light teal for traditional
    
    ax4.set_title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comprehensive comparison saved to: {OUTPUT_DIR / 'comprehensive_model_comparison.png'}")
    
    # Create individual IndoBERT performance plot
    if len(indobert_df) > 0:
        create_indobert_performance_plot(indobert_df)
    
    return comparison_df

def create_indobert_performance_plot(indobert_df):
    """Create detailed IndoBERT performance visualization."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('IndoBERT Learning Rate Optimization Results', fontsize=14, fontweight='bold')
    
    # Filter successful experiments - handle different column structures
    if 'success' in indobert_df.columns:
        successful = indobert_df[indobert_df['success'] == True]
    else:
        # Assume all are successful if no success column
        successful = indobert_df.copy()
    
    if len(successful) == 0:
        print("No successful IndoBERT experiments to visualize")
        return
    
    # Check available columns and use appropriate ones
    if 'learning_rate' in successful.columns:
        learning_rates = successful['learning_rate'].values
    elif 'param_value' in successful.columns:
        learning_rates = successful['param_value'].values
    else:
        print("No learning rate column found in IndoBERT data")
        return
    
    accuracies = successful['accuracy_mean'].values * 100
    f1_scores = successful['f1_mean'].values * 100
    training_times = successful['training_duration'].values / 3600  # Convert to hours
    
    # 1. Performance vs Learning Rate
    ax1 = axes[0]
    ax1.plot(learning_rates, accuracies, 'o-', color='#FF6B6B', linewidth=3, markersize=8, label='Accuracy')
    ax1.plot(learning_rates, f1_scores, 's-', color='#4ECDC4', linewidth=3, markersize=8, label='F1-Score')
    ax1.set_title('Performance vs Learning Rate', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Learning Rate', fontsize=12)
    ax1.set_ylabel('Score (%)', fontsize=12)
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for lr, acc, f1 in zip(learning_rates, accuracies, f1_scores):
        ax1.annotate(f'{acc:.2f}%', (lr, acc), xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax1.annotate(f'{f1:.2f}%', (lr, f1), xytext=(5, -15), textcoords='offset points', fontsize=9)
    
    # 2. Training Time vs Learning Rate
    ax2 = axes[1]
    bars = ax2.bar(learning_rates, training_times, color='#95A5A6', alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Training Time vs Learning Rate', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Learning Rate', fontsize=12)
    ax2.set_ylabel('Training Time (hours)', fontsize=12)
    ax2.set_xscale('log')
    
    # Add value labels
    for bar, time in zip(bars, training_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time:.1f}h', ha='center', va='bottom', fontweight='bold')
    
    # 3. Performance Summary Bar Chart
    ax3 = axes[2]
    x_pos = range(len(learning_rates))
    width = 0.35
    
    bars1 = ax3.bar([x - width/2 for x in x_pos], accuracies, width, 
                   label='Accuracy', color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax3.bar([x + width/2 for x in x_pos], f1_scores, width,
                   label='F1-Score', color='#4ECDC4', alpha=0.8, edgecolor='black')
    
    ax3.set_title('Performance Comparison by Learning Rate', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Learning Rate', fontsize=12)
    ax3.set_ylabel('Score (%)', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{lr:.0e}' for lr in learning_rates])
    ax3.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'indobert_learning_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ IndoBERT analysis saved to: {OUTPUT_DIR / 'indobert_learning_rate_analysis.png'}")

def main():
    """Main function to create all comparison visualizations."""
    
    print("🚀 Creating comprehensive model comparison visualizations...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Create comprehensive comparison
    comparison_df = create_comprehensive_comparison()
    
    print("\n📊 Comparison Results Summary:")
    print("=" * 60)
    for _, row in comparison_df.iterrows():
        model_name = 'IndoBERT' if row['model'] == 'indobert' else row['model'].upper()
        print(f"{model_name:12} | Accuracy: {row['accuracy_mean']*100:6.2f}% | F1: {row['f1_mean']*100:6.2f}% | Time: {row['training_duration']/3600:5.1f}h")
    print("=" * 60)
    
    print(f"\n✅ All visualizations saved to: {OUTPUT_DIR}")
    print("📁 Files created:")
    print(f"  - comprehensive_model_comparison.png")
    if (OUTPUT_DIR / "indobert_learning_rate_analysis.png").exists():
        print(f"  - indobert_learning_rate_analysis.png")

if __name__ == "__main__":
    main()