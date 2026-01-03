"""
Robust IndoBERT Hyperparameter Experiments with Crash Recovery

This enhanced version includes better error handling, progress tracking, and recovery from interruptions.
"""

import os
import sys
import time
import json
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import logging
import signal
import argparse

# =========================
# CONFIGURATION
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "preprocessed_data_FINAL_FINAL.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "indobert_experiments")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "indobert_experiments_results.csv")
PROGRESS_FILE = os.path.join(RESULTS_DIR, "progress.json")

# Model configuration
MODEL_NAME = "indobenchmark/indobert-base-p1"
TEXT_COL = "text_clean"
LABEL_COL = "label_encoded"

# Hyperparameter grid for learning rates
LEARNING_RATES = [1e-5, 2e-5, 3e-5, 5e-5]

# Fixed hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
WEIGHT_DECAY = 0.01
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "experiments.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global shutdown_requested
    logger.info(f"Received signal {signum}. Graceful shutdown requested.")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =========================
# DATA LOADING
# =========================

def load_labeled_data(csv_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    """Load and validate the dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    logger.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate columns
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found. Available: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Available: {list(df.columns)}")
    
    # Clean and prepare data
    df = df[[text_col, label_col]].dropna()
    df = df.rename(columns={text_col: "text", label_col: "label"})
    
    # Filter valid labels
    df = df[df["label"].isin([0, 1])]
    logger.info(f"Loaded {len(df)} valid samples")
    
    return df.reset_index(drop=True)

# =========================
# MODEL TRAINING
# =========================

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize text data."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
        "f1": f1_score(labels, preds, average="weighted"),
    }

def save_progress(progress: Dict):
    """Save experiment progress to file."""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Could not save progress: {e}")

def load_progress() -> Dict:
    """Load experiment progress from file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load progress: {e}")
    return {}

def train_indobert_with_lr(learning_rate: float, df: pd.DataFrame, experiment_id: str) -> Dict:
    """Train IndoBERT with a specific learning rate."""
    
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    
    logger.info(f"Starting experiment {experiment_id} with learning rate {learning_rate}")
    
    try:
        # Check for shutdown request
        if shutdown_requested:
            logger.info("Shutdown requested, stopping experiment")
            return {
                "experiment_id": experiment_id,
                "model": "indobert",
                "learning_rate": learning_rate,
                "success": False,
                "error": "Shutdown requested",
                "timestamp": timestamp,
                "training_duration": time.time() - start_time,
            }
        
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and model
        logger.info(f"Loading IndoBERT: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.to(device)
        
        # Prepare dataset
        logger.info("Preparing dataset...")
        dataset = Dataset.from_pandas(df)
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer, MAX_LENGTH),
            batched=True,
            remove_columns=["text"],
        )
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        
        # Split dataset
        split_dataset = tokenized_dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
        
        logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(RESULTS_DIR, experiment_id),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY,
            logging_dir=os.path.join(RESULTS_DIR, experiment_id, "logs"),
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=1,
            seed=RANDOM_SEED,
            report_to="none",  # Disable wandb/tensorboard
        )
        
        # Trainer
        logger.info("Creating trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Train with progress monitoring
        logger.info(f"Training with learning rate {learning_rate}...")
        
        # Custom training loop with shutdown checking
        for epoch in range(EPOCHS):
            if shutdown_requested:
                logger.info("Shutdown requested during training")
                break
            
            logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
            train_result = trainer.train(resume_from_checkpoint=None if epoch == 0 else None)
            
            # Check shutdown after each epoch
            if shutdown_requested:
                logger.info("Shutdown requested after epoch")
                break
        
        if shutdown_requested:
            return {
                "experiment_id": experiment_id,
                "model": "indobert",
                "learning_rate": learning_rate,
                "success": False,
                "error": "Shutdown requested during training",
                "timestamp": timestamp,
                "training_duration": time.time() - start_time,
            }
        
        # Evaluate
        logger.info("Evaluating model...")
        eval_result = trainer.evaluate()
        
        # Get detailed predictions for confusion matrix
        predictions = trainer.predict(val_dataset)
        preds = predictions.predictions.argmax(-1)
        labels = predictions.label_ids
        
        # Calculate detailed metrics
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="weighted")
        recall = recall_score(labels, preds, average="weighted")
        f1 = f1_score(labels, preds, average="weighted")
        
        # Generate classification report
        class_report = classification_report(
            labels, preds, 
            target_names=["FAKTA(0)", "HOAX(1)"], 
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(labels, preds)
        
        training_duration = time.time() - start_time
        
        # Compile results
        results = {
            "experiment_id": experiment_id,
            "model": "indobert",
            "learning_rate": learning_rate,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "success": True,
            "timestamp": timestamp,
            "training_duration": training_duration,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy_class_0": class_report["FAKTA(0)"]["f1-score"],
            "accuracy_class_1": class_report["HOAX(1)"]["f1-score"],
            "confusion_matrix": conf_matrix.tolist(),
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "eval_loss": eval_result.get("eval_loss", 0),
            "n_train_samples": len(train_dataset),
            "n_val_samples": len(val_dataset),
            "device": str(device),
        }
        
        logger.info(f"Experiment {experiment_id} completed successfully")
        logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {str(e)}")
        return {
            "experiment_id": experiment_id,
            "model": "indobert",
            "learning_rate": learning_rate,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "training_duration": time.time() - start_time,
        }

# =========================
# EXPERIMENT MANAGEMENT
# =========================

def check_existing_experiment(experiment_id: str) -> Dict:
    """Check if an experiment has already been completed successfully."""
    
    # Check if results CSV exists and contains this experiment
    if os.path.exists(RESULTS_CSV):
        try:
            df = pd.read_csv(RESULTS_CSV)
            existing = df[df['experiment_id'] == experiment_id]
            if not existing.empty and existing.iloc[0].get('success', False):
                logger.info(f"Experiment {experiment_id} already completed successfully")
                return existing.iloc[0].to_dict()
        except Exception as e:
            logger.warning(f"Could not check existing CSV: {e}")
    
    # Check individual experiment directory
    exp_dir = os.path.join(RESULTS_DIR, experiment_id)
    details_file = os.path.join(exp_dir, "experiment_details.json")
    
    if os.path.exists(details_file):
        try:
            with open(details_file, 'r') as f:
                result = json.load(f)
                if result.get('success', False):
                    logger.info(f"Experiment {experiment_id} already completed successfully")
                    return result
        except Exception as e:
            logger.warning(f"Could not read details for {experiment_id}: {e}")
    
    return None

def run_indobert_experiments(df: pd.DataFrame) -> List[Dict]:
    """Run all IndoBERT experiments with different learning rates."""
    
    results = []
    progress = load_progress()
    
    # Load existing results if any
    if os.path.exists(RESULTS_CSV):
        try:
            df_existing = pd.read_csv(RESULTS_CSV)
            results = df_existing.to_dict('records')
            logger.info(f"Loaded {len(results)} existing experiment results")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
    
    # Track progress
    total_experiments = len(LEARNING_RATES)
    completed_experiments = len([r for r in results if r.get('success', False)])
    
    logger.info(f"Progress: {completed_experiments}/{total_experiments} experiments completed")
    
    for i, lr in enumerate(LEARNING_RATES):
        if shutdown_requested:
            logger.info("Shutdown requested, stopping experiment runner")
            break
        
        experiment_id = f"indobert_lr_{lr:.0e}".replace("-", "m").replace(".", "_")
        
        # Check if this experiment is already completed
        existing_result = check_existing_experiment(experiment_id)
        if existing_result:
            logger.info(f"Skipping completed experiment: {experiment_id}")
            # Make sure this result is in our results list
            if not any(r.get('experiment_id') == experiment_id for r in results):
                results.append(existing_result)
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment {i+1}/{len(LEARNING_RATES)}: {experiment_id}")
        logger.info(f"Learning rate: {lr:.0e}")
        logger.info(f"Progress: {completed_experiments + 1}/{total_experiments}")
        logger.info(f"{'='*60}")
        
        # Save progress before starting
        progress['current_experiment'] = experiment_id
        progress['learning_rate'] = lr
        progress['status'] = 'running'
        progress['timestamp'] = datetime.now().isoformat()
        save_progress(progress)
        
        result = train_indobert_with_lr(lr, df, experiment_id)
        results.append(result)
        
        # Update progress
        if result.get('success', False):
            completed_experiments += 1
            progress['completed_experiments'] = completed_experiments
            progress['last_successful_experiment'] = experiment_id
        
        progress['status'] = 'completed' if i == len(LEARNING_RATES) - 1 else 'paused'
        save_progress(progress)
        
        # Save intermediate results
        save_results(results)
        
        # Check for shutdown
        if shutdown_requested:
            logger.info("Experiment interrupted, progress saved")
            break
        
        # Brief pause between experiments
        if i < len(LEARNING_RATES) - 1:
            logger.info("Taking a short break before next experiment...")
            time.sleep(30)
    
    return results

def save_results(results: List[Dict]):
    """Save experiment results to CSV."""
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save to CSV
    df_results.to_csv(RESULTS_CSV, index=False)
    logger.info(f"Results saved to {RESULTS_CSV}")
    
    # Also save individual experiment details
    for result in results:
        if result.get("success", False):
            experiment_id = result["experiment_id"]
            details_path = os.path.join(RESULTS_DIR, experiment_id, "experiment_details.json")
            
            try:
                os.makedirs(os.path.dirname(details_path), exist_ok=True)
                with open(details_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Could not save details for {experiment_id}: {e}")

def create_summary_report(results: List[Dict]):
    """Create a summary report of all experiments."""
    
    successful_results = [r for r in results if r.get("success", False)]
    
    if not successful_results:
        logger.warning("No successful experiments to summarize")
        return
    
    # Find best model
    best_result = max(successful_results, key=lambda x: x["f1"])
    
    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(LEARNING_RATES),
        "completed_experiments": len(successful_results),
        "best_model": {
            "experiment_id": best_result["experiment_id"],
            "learning_rate": best_result["learning_rate"],
            "accuracy": best_result["accuracy"],
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "f1": best_result["f1"],
            "training_duration": best_result["training_duration"],
        },
        "all_results": [
            {
                "learning_rate": r["learning_rate"],
                "accuracy": r["accuracy"],
                "f1": r["f1"],
                "training_duration": r["training_duration"],
            }
            for r in successful_results
        ]
    }
    
    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Summary report saved to {summary_path}")
    logger.info(f"Best model: {best_result['experiment_id']} with F1={best_result['f1']:.4f}")

# =========================
# MAIN FUNCTION
# =========================

def main():
    """Main function to run all IndoBERT experiments."""
    
    parser = argparse.ArgumentParser(description="Run IndoBERT hyperparameter experiments")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--check-status", action="store_true", help="Check experiment status only")
    args = parser.parse_args()
    
    logger.info("Starting robust IndoBERT hyperparameter experiments")
    logger.info(f"Learning rates to test: {LEARNING_RATES}")
    logger.info(f"Shutdown handling: Enabled (Ctrl+C to stop gracefully)")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if args.check_status:
        # Just check status and exit
        logger.info("Checking experiment status only...")
        # Import and run status check
        import subprocess
        subprocess.run([sys.executable, "resume_indobert_experiments.py"])
        return
    
    try:
        # Load data
        logger.info("Loading dataset...")
        df = load_labeled_data(DATA_PATH, TEXT_COL, LABEL_COL)
        
        if len(df) == 0:
            logger.error("No valid data loaded")
            return
        
        # Run experiments
        logger.info("Starting experiments...")
        results = run_indobert_experiments(df)
        
        # Create summary
        create_summary_report(results)
        
        if shutdown_requested:
            logger.info("\n" + "="*60)
            logger.info("EXPERIMENTS INTERRUPTED - Progress saved!")
            logger.info("Run again to resume from where you left off.")
            logger.info("="*60)
        else:
            logger.info("\n" + "="*60)
            logger.info("All experiments completed successfully!")
            logger.info(f"Results saved to: {RESULTS_CSV}")
            logger.info(f"Experiment details saved to: {RESULTS_DIR}")
            logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Experiment runner failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()