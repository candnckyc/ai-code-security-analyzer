"""
Model Training Script - FULL DATASET VERSION
Optimized for training on large Devign dataset
"""

import os
import torch
import argparse
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 9800X3D OPTIMIZATION
torch.set_num_threads(14)  # 16 thread'den 14'ü kullan
os.environ["OMP_NUM_THREADS"] = "14"
os.environ["MKL_NUM_THREADS"] = "14"


# Configuration
MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512
BATCH_SIZE = 16  # Increased for full dataset
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5  # More epochs for large dataset
OUTPUT_DIR = "models/finetuned/codebert-security-full"

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

def plot_confusion_matrix(trainer, dataset, output_dir):
    """Plot confusion matrix"""
    print("\nGenerating confusion matrix...")
    
    # Get predictions
    predictions = trainer.predict(dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Safe', 'Vulnerable'],
                yticklabels=['Safe', 'Vulnerable'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {output_dir}/confusion_matrix.png")
    plt.close()

def train_model(dataset_path, args):
    """
    Main training function
    
    Args:
        dataset_path: Path to processed dataset
        args: Command line arguments
    """
    print("=" * 60)
    print("CodeBERT Fine-tuning - FULL DATASET")
    print("=" * 60)
    
    # Step 1: Load dataset
    print("\n[1/7] Loading dataset...")
    dataset = load_from_disk(dataset_path)
    
    train_size = len(dataset['train'])
    val_size = len(dataset['validation'])
    test_size = len(dataset['test'])
    
    print(f"✓ Train samples: {train_size:,}")
    print(f"✓ Validation samples: {val_size:,}")
    print(f"✓ Test samples: {test_size:,}")
    print(f"✓ TOTAL: {train_size + val_size + test_size:,}")
    
    # Check distribution
    train_labels = [ex['target'] for ex in dataset['train']]
    print(f"\nLabel distribution in training:")
    print(f"  Safe (0): {train_labels.count(0):,} ({train_labels.count(0)/len(train_labels)*100:.1f}%)")
    print(f"  Vulnerable (1): {train_labels.count(1):,} ({train_labels.count(1)/len(train_labels)*100:.1f}%)")
    
    # Step 2: Load tokenizer and model
    print("\n[2/7] Loading tokenizer and model...")
    print(f"Model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    print(f"✓ Model loaded with {model.num_parameters():,} parameters")
    
    # Step 3: Tokenize dataset
    print("\n[3/7] Tokenizing dataset (this may take 5-10 minutes)...")
    
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        desc="Tokenizing",
        num_proc=4,  # Parallel processing
        remove_columns=['func']  # Remove to save memory
    )
    
    # Rename target to labels
    tokenized_dataset = tokenized_dataset.rename_column("target", "labels")

    def fix_label_type(example):
        example['labels'] = int(example['labels'])
        return example

    print("\nFixing label types (boolean → int)...")
    tokenized_dataset = tokenized_dataset.map(fix_label_type, batched=False)
    # Cast column dtype to int64 (Long) for PyTorch cross_entropy
    from datasets import Value
    tokenized_dataset = tokenized_dataset.cast_column("labels", Value("int64"))
    print("✓ Labels fixed!")
    
    # Set format for PyTorch
    tokenized_dataset.set_format(
        'torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )
    
    print("✓ Tokenization complete!")
    
    # Step 4: Setup training arguments
    print("\n[4/7] Setting up training...")
    
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs if args.epochs else NUM_EPOCHS,
        per_device_train_batch_size=args.batch_size if args.batch_size else BATCH_SIZE,
        per_device_eval_batch_size=args.batch_size if args.batch_size else BATCH_SIZE,
        learning_rate=args.learning_rate if args.learning_rate else LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=3,  # Keep only best 3 checkpoints
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,  # Parallel data loading
        gradient_accumulation_steps=2,  # Effective batch size = 16*2 = 32
    )
    
    print(f"✓ Training for {training_args.num_train_epochs} epochs")
    print(f"✓ Batch size: {training_args.per_device_train_batch_size}")
    print(f"✓ Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"✓ Learning rate: {training_args.learning_rate}")
    print(f"✓ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: Training on CPU will be VERY slow (hours/days)!")
        print("   Consider using Google Colab GPU for faster training.")
        response = input("\n   Continue with CPU training? (yes/no): ")
        if response.lower() != 'yes':
            print("Training cancelled. Use Google Colab for GPU training.")
            return None, None
    
    # Step 5: Create trainer
    print("\n[5/7] Creating trainer...")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print("✓ Trainer ready!")
    
    # Step 6: Train!
    print("\n[6/7] Starting training...")
    print("=" * 60)
    print("This will take several hours on CPU, ~30-60 min on GPU")
    print("=" * 60)
    
    train_result = trainer.train()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    # Save model
    print("\nSaving model...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print(f"✓ Model saved to: {output_dir}/final")
    
    # Step 7: Evaluate on test set
    print("\n[7/7] Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset['test'])
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test Accuracy:  {test_results['eval_accuracy']:.4f} ({test_results['eval_accuracy']*100:.2f}%)")
    print(f"Test Precision: {test_results['eval_precision']:.4f}")
    print(f"Test Recall:    {test_results['eval_recall']:.4f}")
    print(f"Test F1 Score:  {test_results['eval_f1']:.4f}")
    
    # Save results
    results = {
        'train_loss': float(train_result.training_loss),
        'train_samples': train_size,
        'test_accuracy': float(test_results['eval_accuracy']),
        'test_precision': float(test_results['eval_precision']),
        'test_recall': float(test_results['eval_recall']),
        'test_f1': float(test_results['eval_f1']),
        'num_epochs': training_args.num_train_epochs,
        'batch_size': training_args.per_device_train_batch_size,
        'learning_rate': training_args.learning_rate,
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}/results.json")
    
    # Plot training history
    print("\nGenerating training plots...")
    plot_training_history(trainer, output_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(trainer, tokenized_dataset['test'], output_dir)
    
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss plot
    if train_loss:
        steps = range(len(train_loss))
        axes[0, 0].plot(steps, train_loss, label='Train Loss', marker='o', markersize=3)
        axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Eval loss plot
    if eval_loss:
        epochs = range(1, len(eval_loss) + 1)
        axes[0, 1].plot(epochs, eval_loss, label='Validation Loss', 
                       marker='o', color='orange', linewidth=2)
        axes[0, 1].set_title('Validation Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy plot
    if eval_accuracy:
        epochs = range(1, len(eval_accuracy) + 1)
        axes[1, 0].plot(epochs, eval_accuracy, label='Validation Accuracy', 
                       marker='o', color='green', linewidth=2)
        axes[1, 0].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
    
    # F1 score plot
    if eval_f1:
        epochs = range(1, len(eval_f1) + 1)
        axes[1, 1].plot(epochs, eval_f1, label='Validation F1 Score', 
                       marker='o', color='red', linewidth=2)
        axes[1, 1].set_title('Validation F1 Score', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png", dpi=300, bbox_inches='tight')
    print(f"✓ Training plots saved to: {output_dir}/training_history.png")
    plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train CodeBERT on full Devign dataset')
    parser.add_argument('--dataset', type=str,
                       default='data/processed/devign_full_clean',
                       help='Path to processed dataset')
    parser.add_argument('--output-dir', type=str,
                       default=OUTPUT_DIR,
                       help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"ERROR: Processed dataset not found at {args.dataset}")
        print("\nPlease preprocess the dataset first:")
        print("  python prepare_data_full.py")
        return
    
    # Train model
    trainer, results = train_model(args.dataset, args)
    
    if trainer and results:
        print("\n" + "=" * 60)
        print("✓ TRAINING PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"\nModel saved at: {args.output_dir}/final")
        print(f"Results saved at: {args.output_dir}/results.json")
        print(f"Plots saved at: {args.output_dir}/training_history.png")
        print(f"Confusion matrix: {args.output_dir}/confusion_matrix.png")
        
        print("\n" + "=" * 60)
        print("COMPARE WITH INTERIM REPORT:")
        print("=" * 60)
        print(f"Interim (sample dataset): ~50% accuracy")
        print(f"Current (full dataset):   {results['test_accuracy']*100:.1f}% accuracy")
        print(f"Improvement: +{(results['test_accuracy']-0.5)*100:.1f} percentage points!")
        
        print("\nYou can now use this model for inference!")
        print("  python src/model/test_model.py --model {}/final".format(args.output_dir))

if __name__ == "__main__":
    main()