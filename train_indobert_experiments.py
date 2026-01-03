"""
IndoBERT Hyperparameter Experiments - Learning Rate Grid Search

This script runs IndoBERT experiments with different learning rates to find optimal hyperparameters.
It follows the same pattern as the comprehensive experiments for traditional ML models.

Usage:
    python train_indobert_experiments.py

Outputs:
    - indobert_experiments_results.csv: Results of all experiments
    - indobert_experiments/individual model checkpoints
"""

import os
import sys
import time
import json
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Tuple
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

# =========================
# CONFIGURATION
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "preprocessed_data_FINAL_FINAL.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "indobert_experiments")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "indobert_experiments_results.csv")

# Model configuration
MODEL_NAME = "indobenchmark/indobert-base-p1"
TEXT_COL = "text_clean"
LABEL_COL = "label_encoded"

# Hyperparameter grid for learning rates
LEARNING_RATES = [1e-5, 2e-5, 3e-5, 5e-5]  # 0.00001, 0.00002, 0.00003, 0.00005

# Fixed hyperparameters (based on current best settings)
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
WEIGHT_DECAY = 0.01
MAX_FEATURES = 10000  # For consistency with TF-IDF experiments

# Training configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def train_indobert_with_lr(learning_rate: float, df: pd.DataFrame, experiment_id: str) -> Dict:
    """Train IndoBERT with a specific learning rate."""
    
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    
    logger.info(f"Starting experiment {experiment_id} with learning rate {learning_rate}")
    
    try:
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.to(device)
        
        # Prepare dataset
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
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Train
        logger.info(f"Training with learning rate {learning_rate}...")
        train_result = trainer.train()
        
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
# EXPERIMENT RUNNER
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
    
    # Load existing results if any
    if os.path.exists(RESULTS_CSV):
        try:
            df_existing = pd.read_csv(RESULTS_CSV)
            results = df_existing.to_dict('records')
            logger.info(f"Loaded {len(results)} existing experiment results")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
    
    for i, lr in enumerate(LEARNING_RATES):
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
        logger.info(f"{'='*60}")
        
        result = train_indobert_with_lr(lr, df, experiment_id)
        results.append(result)
        
        # Save intermediate results
        save_results(results)
        
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
            
            with open(details_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)

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
        "total_experiments": len(results),
        "successful_experiments": len(successful_results),
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
    
    logger.info("Starting IndoBERT hyperparameter experiments")
    logger.info(f"Learning rates to test: {LEARNING_RATES}")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
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