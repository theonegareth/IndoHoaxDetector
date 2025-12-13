#!/usr/bin/env python3
"""
Quick test script for the comprehensive experiment framework.

This script runs a small subset of experiments to validate the framework
before running the full comprehensive experiments.

Usage:
    python test_comprehensive_framework.py
"""

import os
import sys
import subprocess
import time

# Add the script directory to the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

def test_single_experiment():
    """Test a single experiment to ensure the framework works."""
    
    print("Testing single Logistic Regression experiment...")
    
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "train_logreg.py"),
        "--c_value", "1.0",
        "--max_features", "1000",
        "--ngram_min", "1",
        "--ngram_max", "1",
        "--output_dir", "test_framework_output"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        print("✅ Single experiment test PASSED")
        print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Single experiment test FAILED")
        print("Error:", e.stderr)
        return False
    except subprocess.TimeoutExpired:
        print("❌ Single experiment test TIMEOUT")
        return False

def test_comprehensive_framework():
    """Test the comprehensive framework with a small subset."""
    
    print("Testing comprehensive framework with small subset...")
    
    # Create a temporary config for testing
    test_config = {
        "models": ["logreg"],
        "tfidf_max_features": [1000, 3000],
        "tfidf_ngram_ranges": [(1, 1), (1, 2)],
        "model_params": {
            "logreg": {"param_name": "c_value", "values": [0.1, 1.0]}
        }
    }
    
    # Calculate expected experiments
    num_experiments = (
        len(test_config["models"]) * 
        len(test_config["tfidf_max_features"]) * 
        len(test_config["tfidf_ngram_ranges"]) * 
        len(test_config["model_params"]["logreg"]["values"])
    )
    
    print(f"Expected experiments: {num_experiments}")
    
    # Test with limited experiments
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "run_comprehensive_experiments.py"),
        "--models", "logreg",
        "--output_dir", "test_comprehensive_output",
        "--max_parallel", "1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        if result.returncode == 0:
            print("✅ Comprehensive framework test PASSED")
            print("Framework is ready for full experiments!")
            return True
        else:
            print("❌ Comprehensive framework test FAILED")
            print("Error:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Comprehensive framework test TIMEOUT")
        return False

def main():
    print("="*60)
    print("COMPREHENSIVE EXPERIMENT FRAMEWORK TEST")
    print("="*60)
    
    # Test 1: Single experiment
    if not test_single_experiment():
        print("\n❌ Framework test FAILED at single experiment stage")
        return False
    
    print("\n" + "-"*60)
    
    # Test 2: Comprehensive framework (small subset)
    if not test_comprehensive_framework():
        print("\n❌ Framework test FAILED at comprehensive stage")
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - FRAMEWORK READY!")
    print("="*60)
    print("\nYou can now run the full comprehensive experiments:")
    print("python run_comprehensive_experiments.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)