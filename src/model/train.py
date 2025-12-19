"""
Model Training Script
Fine-tunes CodeBERT for vulnerability detection
"""

import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import json

# Configuration
MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
OUTPUT_DIR = "models/finetuned/codebert-security"

def tokenize_function(examples, tokenizer):
    """Tokenize code samples"""
    return tokenizer(
        examples['func'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length'
    )

def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(dataset_path):
    """
    Main training function
    
    Args:
        dataset_path: Path to processed dataset
    """
    print("=" * 60)
    print("CodeBERT Fine-tuning for Vulnerability Detection")
    print("=" * 60)
    
    # Step 1: Load dataset
    print("\n[1/6] Loading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"✓ Train samples: {len(dataset['train'])}")
    print(f"✓ Validation samples: {len(dataset['validation'])}")
    print(f"✓ Test samples: {len(dataset['test'])}")
    
    # Check distribution
    train_labels = [ex['target'] for ex in dataset['train']]
    print(f"\nLabel distribution in training:")
    print(f"  Safe (0): {train_labels.count(0)}")
    print(f"  Vulnerable (1): {train_labels.count(1)}")
    
    # Step 2: Load tokenizer and model
    print("\n[2/6] Loading tokenizer and model...")
    print(f"Model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    print(f"✓ Model loaded with {model.num_parameters():,} parameters")
    
    # Step 3: Tokenize dataset
    print("\n[3/6] Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        desc="Tokenizing"
    )
    
    # Rename target to labels for training
    tokenized_dataset = tokenized_dataset.rename_column("target", "labels")
    
    # Set format for PyTorch
    tokenized_dataset.set_format(
        'torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )
    
    print("✓ Tokenization complete!")
    
    # Step 4: Setup training arguments
    print("\n[4/6] Setting up training...")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        report_to="none",  # Disable wandb
        fp16=torch.cuda.is_available(),  # Use fp16 if GPU available
    )
    
    print(f"✓ Training for {NUM_EPOCHS} epochs")
    print(f"✓ Batch size: {BATCH_SIZE}")
    print(f"✓ Learning rate: {LEARNING_RATE}")
    print(f"✓ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Step 5: Create trainer
    print("\n[5/6] Creating trainer...")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("✓ Trainer ready!")
    
    # Step 6: Train!
    print("\n[6/6] Starting training...")
    print("=" * 60)
    
    train_result = trainer.train()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    # Save model
    print("\nSaving model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"✓ Model saved to: {OUTPUT_DIR}/final")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset['test'])
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test Accuracy:  {test_results['eval_accuracy']:.4f}")
    print(f"Test Precision: {test_results['eval_precision']:.4f}")
    print(f"Test Recall:    {test_results['eval_recall']:.4f}")
    print(f"Test F1 Score:  {test_results['eval_f1']:.4f}")
    
    # Save results
    results = {
        'train_loss': train_result.training_loss,
        'test_accuracy': test_results['eval_accuracy'],
        'test_precision': test_results['eval_precision'],
        'test_recall': test_results['eval_recall'],
        'test_f1': test_results['eval_f1'],
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {OUTPUT_DIR}/results.json")
    
    # Plot training history
    print("\nGenerating training plots...")
    plot_training_history(trainer, OUTPUT_DIR)
    
    return trainer, results

def plot_training_history(trainer, output_dir):
    """Plot and save training history"""
    log_history = trainer.state.log_history
    
    # Extract metrics
    train_loss = []
    eval_loss = []
    eval_accuracy = []
    eval_f1 = []
    
    for entry in log_history:
        if 'loss' in entry:
            train_loss.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
        if 'eval_accuracy' in entry:
            eval_accuracy.append(entry['eval_accuracy'])
        if 'eval_f1' in entry:
            eval_f1.append(entry['eval_f1'])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss plot
    if train_loss:
        axes[0, 0].plot(train_loss, label='Train Loss', marker='o')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Eval loss plot
    if eval_loss:
        epochs = range(1, len(eval_loss) + 1)
        axes[0, 1].plot(epochs, eval_loss, label='Validation Loss', 
                       marker='o', color='orange')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Accuracy plot
    if eval_accuracy:
        epochs = range(1, len(eval_accuracy) + 1)
        axes[1, 0].plot(epochs, eval_accuracy, label='Accuracy', 
                       marker='o', color='green')
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim([0, 1])
    
    # F1 score plot
    if eval_f1:
        epochs = range(1, len(eval_f1) + 1)
        axes[1, 1].plot(epochs, eval_f1, label='F1 Score', 
                       marker='o', color='red')
        axes[1, 1].set_title('Validation F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png", dpi=300, bbox_inches='tight')
    print(f"✓ Training plots saved to: {output_dir}/training_history.png")
    plt.close()

def main():
    """Main function"""
    # Check if processed dataset exists
    dataset_path = "data/processed/sample_dataset_clean"
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Processed dataset not found at {dataset_path}")
        print("Please run prepare_data.py first!")
        return
    
    # Train model
    trainer, results = train_model(dataset_path)
    
    print("\n" + "=" * 60)
    print("✓ Training pipeline complete!")
    print("=" * 60)
    print(f"\nModel saved at: {OUTPUT_DIR}/final")
    print(f"Results saved at: {OUTPUT_DIR}/results.json")
    print(f"Plots saved at: {OUTPUT_DIR}/training_history.png")
    print("\nYou can now use this model for inference!")

if __name__ == "__main__":
    main()