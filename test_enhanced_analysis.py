"""
Test script for enhanced error analysis integration.
This script demonstrates how to use the new enhanced error analysis feature.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate_model import run_evaluation

def test_enhanced_analysis():
    """Test the enhanced error analysis feature."""
    print("Testing enhanced error analysis...")
    print("=" * 60)
    
    try:
        # Run evaluation with enhanced analysis
        eval_df = run_evaluation(
            data_path="preprocessed_data_FINAL_FINAL.csv",
            text_col="text_clean",
            label_col="label_encoded",
            model_path="logreg_model.pkl",
            vectorizer_path="tfidf_vectorizer.pkl",
            max_show=3,
            enhanced_analysis=True
        )
        
        print("\n✅ Enhanced error analysis test completed successfully!")
        print(f"   - Evaluated {len(eval_df)} samples")
        print(f"   - Check 'plots/error_analysis/' directory for visualizations")
        print(f"   - Check 'error_analysis_report.html' for HTML report")
        print(f"   - Check 'error_samples.csv' for detailed error samples")
        
        # Show basic stats
        fp = eval_df[(eval_df["true_label"] == 0) & (eval_df["pred_label"] == 1)]
        fn = eval_df[(eval_df["true_label"] == 1) & (eval_df["pred_label"] == 0)]
        print(f"\n   - False Positives: {len(fp)}")
        print(f"   - False Negatives: {len(fn)}")
        print(f"   - Accuracy: {(eval_df['true_label'] == eval_df['pred_label']).mean():.4f}")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  File not found: {e}")
        print("   Make sure you're running from the correct directory with the required files.")
        print("   Required files:")
        print("   - preprocessed_data_FINAL_FINAL.csv")
        print("   - logreg_model.pkl")
        print("   - tfidf_vectorizer.pkl")
    except ImportError as e:
        print(f"\n⚠️  Import error: {e}")
        print("   Install required packages: pip install matplotlib seaborn wordcloud")
    except Exception as e:
        print(f"\n❌ Error during enhanced analysis: {e}")
        import traceback
        traceback.print_exc()

def test_basic_evaluation():
    """Test basic evaluation without enhanced analysis."""
    print("\n" + "=" * 60)
    print("Testing basic evaluation...")
    print("=" * 60)
    
    try:
        eval_df = run_evaluation(
            data_path="preprocessed_data_FINAL_FINAL.csv",
            text_col="text_clean",
            label_col="label_encoded",
            model_path="logreg_model.pkl",
            vectorizer_path="tfidf_vectorizer.pkl",
            max_show=2,
            enhanced_analysis=False
        )
        
        print("\n✅ Basic evaluation test completed successfully!")
        print(f"   - Evaluated {len(eval_df)} samples")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  File not found: {e}")
    except Exception as e:
        print(f"\n❌ Error during basic evaluation: {e}")

if __name__ == "__main__":
    print("IndoHoaxDetector Enhanced Error Analysis Test")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        "preprocessed_data_FINAL_FINAL.csv",
        "logreg_model.pkl",
        "tfidf_vectorizer.pkl",
        "evaluate_model.py",
        "error_analysis.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
        print("   Some tests may fail. Please ensure all required files are present.")
        print("   You may need to adjust paths or download the dataset.")
    
    # Run tests
    test_basic_evaluation()
    test_enhanced_analysis()
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("\nNext steps:")
    print("1. Run 'python evaluate_model.py --enhanced-analysis' for full analysis")
    print("2. Check the generated plots in 'plots/error_analysis/'")
    print("3. Open 'error_analysis_report.html' for interactive analysis")
    print("4. Review 'error_samples.csv' for detailed error categorization")