"""
Enhanced Error Analysis for IndoHoaxDetector

This module provides advanced error analysis capabilities including:
1. Error categorization by type (sensationalism, neutral tone, domain shift, etc.)
2. Confidence-based error analysis
3. Text length analysis
4. Visualization generation
5. Detailed error reporting

Usage:
    from error_analysis import ErrorAnalyzer
    analyzer = ErrorAnalyzer(df_with_predictions)
    analyzer.analyze()
    analyzer.generate_report("error_report.html")
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ErrorAnalyzer:
    """Enhanced error analysis for classification models."""
    
    def __init__(self, df: pd.DataFrame, text_col: str = "text", 
                 true_label_col: str = "true_label", pred_label_col: str = "pred_label",
                 confidence_col: str = "confidence"):
        """
        Initialize with DataFrame containing predictions.
        
        Args:
            df: DataFrame with columns for text, true labels, predicted labels, and confidence
            text_col: Column name for text content
            true_label_col: Column name for true labels (0=FAKTA, 1=HOAX)
            pred_label_col: Column name for predicted labels (0=FAKTA, 1=HOAX)
            confidence_col: Column name for prediction confidence scores
        """
        self.df = df.copy()
        self.text_col = text_col
        self.true_label_col = true_label_col
        self.pred_label_col = pred_label_col
        self.confidence_col = confidence_col
        
        # Add error type column
        self.df['error_type'] = self._determine_error_type()
        
        # Calculate text length
        self.df['text_length'] = self.df[text_col].astype(str).apply(len)
        
        # Error flags
        self.df['is_correct'] = self.df[true_label_col] == self.df[pred_label_col]
        self.df['is_fp'] = (self.df[true_label_col] == 0) & (self.df[pred_label_col] == 1)
        self.df['is_fn'] = (self.df[true_label_col] == 1) & (self.df[pred_label_col] == 0)
        
        # Confidence bins
        self.df['confidence_bin'] = pd.cut(
            self.df[confidence_col], 
            bins=[0, 0.5, 0.7, 0.9, 0.95, 1.0],
            labels=['0-0.5', '0.5-0.7', '0.7-0.9', '0.9-0.95', '0.95-1.0']
        )
        
        # Initialize results storage
        self.results = {}
        
    def _determine_error_type(self) -> pd.Series:
        """Categorize errors based on text characteristics."""
        error_types = []
        
        for idx, row in self.df.iterrows():
            text = str(row[self.text_col]).lower()
            true_label = row[self.true_label_col]
            pred_label = row[self.pred_label_col]
            
            # If prediction is correct, no error
            if true_label == pred_label:
                error_types.append('correct')
                continue
                
            # Determine error characteristics
            is_fp = (true_label == 0) and (pred_label == 1)  # FAKTA predicted as HOAX
            is_fn = (true_label == 1) and (pred_label == 0)  # HOAX predicted as FAKTA
            
            # Check for sensational language
            sensational_words = ['breaking', 'urgent', 'shocking', 'alert', 'warning', 
                               'emergency', 'crisis', 'exposed', 'revealed', 'secret']
            has_sensational = any(word in text for word in sensational_words)
            
            # Check for neutral/formal language
            formal_words = ['according to', 'report', 'statement', 'official', 
                          'announced', 'confirmed', 'according', 'said']
            has_formal = any(word in text for word in formal_words)
            
            # Check for emotional language
            emotional_words = ['angry', 'happy', 'sad', 'fear', 'scared', 'love', 
                             'hate', 'disgust', 'outrage', 'joy']
            has_emotional = any(word in text for word in emotional_words)
            
            # Check for question marks (indicating uncertainty)
            has_question = '?' in text
            
            # Check for exclamation marks (indicating emphasis)
            has_exclamation = '!' in text
            
            # Text length
            text_len = len(text.split())
            is_short = text_len < 10
            is_long = text_len > 100
            
            # Determine error type based on characteristics
            if is_fp:
                if has_sensational:
                    error_types.append('fp_sensational')
                elif has_emotional:
                    error_types.append('fp_emotional')
                elif has_exclamation:
                    error_types.append('fp_exclamatory')
                elif is_short:
                    error_types.append('fp_short_text')
                else:
                    error_types.append('fp_general')
            elif is_fn:
                if has_formal:
                    error_types.append('fn_formal')
                elif has_question:
                    error_types.append('fn_questioning')
                elif is_long:
                    error_types.append('fn_long_text')
                else:
                    error_types.append('fn_general')
            else:
                error_types.append('correct')
                
        return pd.Series(error_types, index=self.df.index)
    
    def analyze(self) -> Dict:
        """Perform comprehensive error analysis."""
        print("=" * 60)
        print("ENHANCED ERROR ANALYSIS")
        print("=" * 60)
        
        # Basic metrics
        total = len(self.df)
        correct = self.df['is_correct'].sum()
        fp = self.df['is_fp'].sum()
        fn = self.df['is_fn'].sum()
        
        accuracy = correct / total if total > 0 else 0
        fp_rate = fp / total if total > 0 else 0
        fn_rate = fn / total if total > 0 else 0
        
        self.results['basic_metrics'] = {
            'total_samples': total,
            'correct_predictions': correct,
            'false_positives': fp,
            'false_negatives': fn,
            'accuracy': accuracy,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate
        }
        
        print(f"\n📊 Basic Metrics:")
        print(f"   Total samples: {total}")
        print(f"   Correct predictions: {correct} ({accuracy:.2%})")
        print(f"   False Positives: {fp} ({fp_rate:.2%})")
        print(f"   False Negatives: {fn} ({fn_rate:.2%})")
        
        # Error type distribution
        error_type_counts = self.df[self.df['error_type'] != 'correct']['error_type'].value_counts()
        self.results['error_type_distribution'] = error_type_counts.to_dict()
        
        print(f"\n🔍 Error Type Distribution:")
        for error_type, count in error_type_counts.items():
            percentage = count / (fp + fn) * 100 if (fp + fn) > 0 else 0
            print(f"   {error_type}: {count} ({percentage:.1f}% of errors)")
        
        # Confidence analysis
        print(f"\n🎯 Confidence Analysis:")
        for error_flag, label in [('is_fp', 'False Positives'), ('is_fn', 'False Negatives')]:
            error_df = self.df[self.df[error_flag]]
            if len(error_df) > 0:
                avg_conf = error_df[self.confidence_col].mean()
                median_conf = error_df[self.confidence_col].median()
                high_conf = (error_df[self.confidence_col] > 0.9).sum()
                print(f"   {label}:")
                print(f"     Average confidence: {avg_conf:.3f}")
                print(f"     Median confidence: {median_conf:.3f}")
                print(f"     High confidence (>0.9): {high_conf}/{len(error_df)} ({high_conf/len(error_df):.1%})")
        
        # Text length analysis
        print(f"\n📏 Text Length Analysis:")
        for label in ['correct', 'fp', 'fn']:
            if label == 'correct':
                subset = self.df[self.df['is_correct']]
            elif label == 'fp':
                subset = self.df[self.df['is_fp']]
            else:  # fn
                subset = self.df[self.df['is_fn']]
            
            if len(subset) > 0:
                avg_len = subset['text_length'].mean()
                print(f"   {label.upper()}: Average text length = {avg_len:.1f} chars")
        
        # Word frequency analysis for errors
        print(f"\n🔤 Top Words in Errors:")
        error_texts = self.df[~self.df['is_correct']][self.text_col].astype(str).tolist()
        if error_texts:
            all_words = []
            for text in error_texts:
                words = re.findall(r'\b\w+\b', text.lower())
                all_words.extend(words)
            
            word_counts = Counter(all_words)
            common_words = word_counts.most_common(10)
            print("   Most common words in errors:")
            for word, count in common_words:
                print(f"     '{word}': {count} times")
        
        return self.results
    
    def generate_visualizations(self, save_dir: str = "plots/error_analysis"):
        """Generate and save visualization plots."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. Error type distribution
        error_types = self.df[self.df['error_type'] != 'correct']['error_type'].value_counts()
        axes[0].bar(error_types.index, error_types.values)
        axes[0].set_title('Error Type Distribution')
        axes[0].set_xlabel('Error Type')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. Confidence distribution by prediction correctness
        axes[1].hist([
            self.df[self.df['is_correct']][self.confidence_col],
            self.df[self.df['is_fp']][self.confidence_col],
            self.df[self.df['is_fn']][self.confidence_col]
        ], bins=20, label=['Correct', 'False Positive', 'False Negative'], alpha=0.7)
        axes[1].set_title('Confidence Distribution by Prediction Type')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        # 3. Text length vs confidence scatter
        scatter = axes[2].scatter(
            self.df['text_length'], 
            self.df[self.confidence_col],
            c=self.df['is_correct'].map({True: 'green', False: 'red'}),
            alpha=0.6
        )
        axes[2].set_title('Text Length vs Confidence')
        axes[2].set_xlabel('Text Length (chars)')
        axes[2].set_ylabel('Confidence')
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Correct'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Incorrect')
        ]
        axes[2].legend(handles=legend_elements)
        
        # 4. Confidence bins for errors
        error_df = self.df[~self.df['is_correct']]
        if not error_df.empty:
            conf_bins = error_df['confidence_bin'].value_counts().sort_index()
            axes[3].bar(conf_bins.index.astype(str), conf_bins.values)
            axes[3].set_title('Error Confidence Distribution')
            axes[3].set_xlabel('Confidence Bin')
            axes[3].set_ylabel('Error Count')
            axes[3].tick_params(axis='x', rotation=45)
        
        # 5. Error rate by text length quartile
        self.df['length_quartile'] = pd.qcut(self.df['text_length'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        error_rate_by_quartile = self.df.groupby('length_quartile').apply(
            lambda x: (len(x[~x['is_correct']]) / len(x)) * 100
        )
        axes[4].bar(error_rate_by_quartile.index.astype(str), error_rate_by_quartile.values)
        axes[4].set_title('Error Rate by Text Length Quartile')
        axes[4].set_xlabel('Text Length Quartile')
        axes[4].set_ylabel('Error Rate (%)')
        
        # 6. FP vs FN by error type
        fp_counts = self.df[self.df['is_fp']]['error_type'].value_counts()
        fn_counts = self.df[self.df['is_fn']]['error_type'].value_counts()
        
        error_types_union = set(fp_counts.index) | set(fn_counts.index)
        x = np.arange(len(error_types_union))
        width = 0.35
        
        axes[5].bar(x - width/2, [fp_counts.get(et, 0) for et in error_types_union], width, label='FP')
        axes[5].bar(x + width/2, [fn_counts.get(et, 0) for et in error_types_union], width, label='FN')
        axes[5].set_title('FP vs FN by Error Type')
        axes[5].set_xlabel('Error Type')
        axes[5].set_ylabel('Count')
        axes[5].set_xticks(x)
        axes[5].set_xticklabels(list(error_types_union), rotation=45, ha='right')
        axes[5].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/error_analysis_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Visualizations saved to {save_dir}/error_analysis_dashboard.png")
        
        # Create word cloud for errors if wordcloud is available
        try:
            from wordcloud import WordCloud
            
            error_texts = ' '.join(self.df[~self.df['is_correct']][self.text_col].astype(str).tolist())
            if error_texts.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(error_texts)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud of Error Texts')
                plt.savefig(f"{save_dir}/error_wordcloud.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Word cloud saved to {save_dir}/error_wordcloud.png")
        except ImportError:
            print("⚠️  WordCloud not installed. Install with: pip install wordcloud")
        
        return save_dir
    
    def generate_report(self, output_path: str = "error_analysis_report.html"):
        """Generate an HTML report with analysis results."""
        from datetime import datetime
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>IndoHoaxDetector Error Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
                .metric {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
                .error-type {{ margin: 5px 0; padding: 8px; background-color: #ecf0f1; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .images {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
                .images img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>IndoHoaxDetector Error Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>📊 Basic Metrics</h2>
            <div class="metric">
                <div>Total Samples: <span class="metric-value">{self.results['basic_metrics']['total_samples']}</span></div>
                <div>Accuracy: <span class="metric-value">{self.results['basic_metrics']['accuracy']:.2%}</span></div>
                <div>False Positives: <span class="metric-value">{self.results['basic_metrics']['false_positives']} ({self.results['basic_metrics']['fp_rate']:.2%})</span></div>
                <div>False Negatives: <span class="metric-value">{self.results['basic_metrics']['false_negatives']} ({self.results['basic_metrics']['fn_rate']:.2%})</span></div>
            </div>
            
            <h2>🔍 Error Type Distribution</h2>
            <table>
                <tr><th>Error Type</th><th>Count</th><th>Percentage of Errors</th></tr>
        """
        
        total_errors = self.results['basic_metrics']['false_positives'] + self.results['basic_metrics']['false_negatives']
        for error_type, count in self.results.get('error_type_distribution', {}).items():
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            html_content += f"<tr><td>{error_type}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html_content += """
            </table>
            
            <h2>📈 Visualizations</h2>
            <div class="images">
                <div>
                    <h3>Error Analysis Dashboard</h3>
                    <img src="plots/error_analysis/error_analysis_dashboard.png" alt="Error Analysis Dashboard" style="width: 600px;">
                </div>
        """
        
        # Check if wordcloud exists
        import os
        if os.path.exists("plots/error_analysis/error_wordcloud.png"):
            html_content += """
                <div>
                    <h3>Word Cloud of Error Texts</h3>
                    <img src="plots/error_analysis/error_wordcloud.png" alt="Error Word Cloud" style="width: 600px;">
                </div>
            """
        
        html_content += """
            </div>
            
            <h2>🎯 Recommendations</h2>
            <ul>
                <li><strong>High-confidence errors:</strong> Review model calibration and consider adjusting decision thresholds</li>
                <li><strong>Sensational language false positives:</strong> Add feature engineering to distinguish sensational but factual news</li>
                <li><strong>Formal language false negatives:</strong> Improve detection of sophisticated hoaxes that mimic official communication</li>
                <li><strong>Short text errors:</strong> Consider minimum length requirements or special handling for short texts</li>
                <li><strong>Domain adaptation:</strong> Fine-tune on examples from error categories with highest error rates</li>
            </ul>
            
            <h2>📋 Sample Errors by Type</h2>
        """
        
        # Add sample errors for each type
        error_types_to_show = ['fp_sensational', 'fp_emotional', 'fn_formal', 'fn_questioning']
        for error_type in error_types_to_show:
            samples = self.df[self.df['error_type'] == error_type].head(3)
            if not samples.empty:
                html_content += f"<h3>{error_type}</h3>"
                for idx, row in samples.iterrows():
                    snippet = str(row[self.text_col])[:200] + "..." if len(str(row[self.text_col])) > 200 else str(row[self.text_col])
                    html_content += f"""
                    <div class="error-type">
                        <strong>Confidence: {row[self.confidence_col]:.3f}</strong><br>
                        {snippet}
                    </div>
                    """
        
        html_content += """
            </body>
            </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✅ HTML report saved to {output_path}")
        return output_path
    
    def export_error_samples(self, output_csv: str = "error_samples.csv"):
        """Export detailed error samples for manual review."""
        error_df = self.df[~self.df['is_correct']].copy()
        
        # Select relevant columns
        export_cols = [self.text_col, self.true_label_col, self.pred_label_col, 
                      self.confidence_col, 'error_type', 'text_length', 'confidence_bin']
        
        # Add true and predicted labels as strings
        error_df['true_label_str'] = error_df[self.true_label_col].map({0: 'FAKTA', 1: 'HOAX'})
        error_df['pred_label_str'] = error_df[self.pred_label_col].map({0: 'FAKTA', 1: 'HOAX'})
        
        export_df = error_df[export_cols + ['true_label_str', 'pred_label_str']]
        export_df.to_csv(output_csv, index=False, encoding='utf-8')
        
        print(f"\n✅ Error samples exported to {output_csv}")
        return output_csv


def analyze_errors_from_evaluation(df_with_predictions: pd.DataFrame, 
                                   text_col: str = "text",
                                   true_label_col: str = "true_label",
                                   pred_label_col: str = "pred_label",
                                   confidence_col: str = "confidence",
                                   generate_plots: bool = True,
                                   output_dir: str = "plots/error_analysis") -> ErrorAnalyzer:
    """
    Convenience function to run complete error analysis.
    
    Args:
        df_with_predictions: DataFrame with predictions from evaluate_model.py
        text_col: Column name for text
        true_label_col: Column name for true labels
        pred_label_col: Column name for predicted labels
        confidence_col: Column name for confidence scores
        generate_plots: Whether to generate visualization plots
        output_dir: Directory to save plots
        
    Returns:
        ErrorAnalyzer instance with analysis results
    """
    print("🔍 Starting enhanced error analysis...")
    
    analyzer = ErrorAnalyzer(
        df=df_with_predictions,
        text_col=text_col,
        true_label_col=true_label_col,
        pred_label_col=pred_label_col,
        confidence_col=confidence_col
    )
    
    # Run analysis
    results = analyzer.analyze()
    
    # Generate visualizations
    if generate_plots:
        analyzer.generate_visualizations(save_dir=output_dir)
    
    # Generate report
    analyzer.generate_report("error_analysis_report.html")
    
    # Export error samples
    analyzer.export_error_samples("error_samples.csv")
    
    print("\n✅ Enhanced error analysis complete!")
    print("   - Visualizations saved to plots/error_analysis/")
    print("   - HTML report: error_analysis_report.html")
    print("   - Error samples: error_samples.csv")
    
    return analyzer


if __name__ == "__main__":
    # Example usage
    print("Error Analysis Module for IndoHoaxDetector")
    print("=" * 50)
    print("\nTo use this module:")
    print("1. First run evaluate_model.py to get predictions")
    print("2. Then import and use:")
    print("""
    from evaluate_model import run_evaluation
    from error_analysis import analyze_errors_from_evaluation
    
    # Get predictions
    eval_df = run_evaluation()
    
    # Run enhanced error analysis
    analyzer = analyze_errors_from_evaluation(eval_df)
    """)