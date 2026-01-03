"""
Resume IndoBERT Hyperparameter Experiments

This script checks for existing experiment results and resumes from where the previous run was interrupted.
"""

import os
import sys
import json
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "indobert_experiments")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "indobert_experiments_results.csv")

LEARNING_RATES = [1e-5, 2e-5, 3e-5, 5e-5]

def check_existing_results() -> List[Dict]:
    """Check for existing experiment results."""
    results = []
    
    if os.path.exists(RESULTS_CSV):
        try:
            df = pd.read_csv(RESULTS_CSV)
            results = df.to_dict('records')
            logger.info(f"Found existing results with {len(results)} experiments")
        except Exception as e:
            logger.warning(f"Could not read existing CSV: {e}")
    
    # Also check individual experiment directories
    for lr in LEARNING_RATES:
        experiment_id = f"indobert_lr_{lr:.0e}".replace("-", "m").replace(".", "_")
        exp_dir = os.path.join(RESULTS_DIR, experiment_id)
        
        if os.path.exists(exp_dir):
            details_file = os.path.join(exp_dir, "experiment_details.json")
            if os.path.exists(details_file):
                try:
                    with open(details_file, 'r') as f:
                        result = json.load(f)
                        # Check if this result is already in our list
                        if not any(r.get('experiment_id') == experiment_id for r in results):
                            results.append(result)
                            logger.info(f"Found individual result for {experiment_id}")
                except Exception as e:
                    logger.warning(f"Could not read details for {experiment_id}: {e}")
    
    return results

def get_completed_learning_rates(results: List[Dict]) -> List[float]:
    """Get list of learning rates that have been successfully completed."""
    completed_lrs = []
    for result in results:
        if result.get('success', False) and 'learning_rate' in result:
            completed_lrs.append(result['learning_rate'])
    return list(set(completed_lrs))  # Remove duplicates

def get_remaining_learning_rates(completed_lrs: List[float]) -> List[float]:
    """Get learning rates that still need to be tested."""
    remaining = [lr for lr in LEARNING_RATES if lr not in completed_lrs]
    return remaining

def create_resume_report(results: List[Dict], completed_lrs: List[float], remaining_lrs: List[float]):
    """Create a report about the current state of experiments."""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(LEARNING_RATES),
        "completed_experiments": len(completed_lrs),
        "remaining_experiments": len(remaining_lrs),
        "completed_learning_rates": completed_lrs,
        "remaining_learning_rates": remaining_lrs,
        "successful_results": []
    }
    
    # Add successful results
    for result in results:
        if result.get('success', False):
            report["successful_results"].append({
                "experiment_id": result.get('experiment_id'),
                "learning_rate": result.get('learning_rate'),
                "accuracy": result.get('accuracy'),
                "f1": result.get('f1'),
                "training_duration": result.get('training_duration'),
            })
    
    # Save report
    report_path = os.path.join(RESULTS_DIR, "resume_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Resume report saved to {report_path}")
    return report

def main():
    """Main function to check experiment status and create resume report."""
    
    logger.info("Checking IndoBERT experiment status...")
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check existing results
    existing_results = check_existing_results()
    
    # Get completed and remaining learning rates
    completed_lrs = get_completed_learning_rates(existing_results)
    remaining_lrs = get_remaining_learning_rates(completed_lrs)
    
    # Create report
    report = create_resume_report(existing_results, completed_lrs, remaining_lrs)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT STATUS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total experiments planned: {report['total_experiments']}")
    logger.info(f"Completed experiments: {report['completed_experiments']}")
    logger.info(f"Remaining experiments: {report['remaining_experiments']}")
    
    if report['successful_results']:
        logger.info("\nCOMPLETED EXPERIMENTS:")
        for result in report['successful_results']:
            logger.info(f"  - LR {result['learning_rate']:.0e}: Accuracy={result['accuracy']:.4f}, F1={result['f1']:.4f}")
    
    if remaining_lrs:
        logger.info(f"\nREMAINING EXPERIMENTS (learning rates): {[f'{lr:.0e}' for lr in remaining_lrs]}")
        logger.info("\nTo continue experiments, run:")
        logger.info(f"python3 train_indobert_experiments.py")
        logger.info("The script will automatically skip completed experiments.")
    else:
        logger.info("\nAll experiments completed!")
        logger.info("Check the results in:")
        logger.info(f"  - {RESULTS_CSV}")
        logger.info(f"  - {RESULTS_DIR}")
    
    logger.info("="*60)
    
    return remaining_lrs

if __name__ == "__main__":
    main()