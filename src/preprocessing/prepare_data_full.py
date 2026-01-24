"""
Data Preparation Script - FULL DATASET VERSION
Handles large Devign dataset (~27K examples)
"""

import os
import sys
import argparse
from datasets import load_from_disk, DatasetDict
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocessing.parser import clean_code

def preprocess_example(example):
    """Preprocess a single example"""
    example['func'] = clean_code(example['func'])
    return example

def prepare_full_dataset(input_path, output_path, batch_size=1000):
    """
    Load full dataset, clean it, and save processed version
    Uses batching for memory efficiency
    
    Args:
        input_path: Path to raw dataset
        output_path: Path to save processed dataset
        batch_size: Process in batches to save memory
    """
    print("=" * 60)
    print("FULL DEVIGN DATASET PREPROCESSING")
    print("=" * 60)
    
    # Step 1: Load raw dataset
    print("\n[1/4] Loading full dataset...")
    dataset = load_from_disk(input_path)
    
    train_size = len(dataset['train'])
    val_size = len(dataset['validation'])
    test_size = len(dataset['test'])
    total_size = train_size + val_size + test_size
    
    print(f"✓ Loaded dataset:")
    print(f"  Train: {train_size:,} samples")
    print(f"  Validation: {val_size:,} samples")
    print(f"  Test: {test_size:,} samples")
    print(f"  TOTAL: {total_size:,} samples")
    
    # Step 2: Show example before processing
    print("\n[2/4] Example BEFORE preprocessing:")
    print("-" * 60)
    example = dataset['train'][0]['func'][:200]
    print(example + "...")
    
    # Step 3: Clean all splits with batching
    print("\n[3/4] Cleaning code samples (this may take 10-20 minutes)...")
    
    print(f"\n  Processing train set ({train_size:,} samples)...")
    train_clean = dataset['train'].map(
        preprocess_example,
        batched=False,
        desc="Cleaning train",
        num_proc=4  # Use 4 CPU cores for parallel processing
    )
    
    print(f"\n  Processing validation set ({val_size:,} samples)...")
    val_clean = dataset['validation'].map(
        preprocess_example,
        batched=False,
        desc="Cleaning validation",
        num_proc=4
    )
    
    print(f"\n  Processing test set ({test_size:,} samples)...")
    test_clean = dataset['test'].map(
        preprocess_example,
        batched=False,
        desc="Cleaning test",
        num_proc=4
    )
    
    # Create cleaned dataset
    cleaned_dataset = DatasetDict({
        'train': train_clean,
        'validation': val_clean,
        'test': test_clean
    })
    
    # Show example after processing
    print("\n  Example AFTER preprocessing:")
    print("-" * 60)
    example_clean = cleaned_dataset['train'][0]['func'][:200]
    print(example_clean + "...")
    
    # Step 4: Save processed dataset
    print("\n[4/4] Saving processed dataset...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_dataset.save_to_disk(output_path)
    print(f"✓ Saved to: {output_path}")
    
    # Statistics
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nProcessed {len(cleaned_dataset['train']):,} train samples")
    print(f"Processed {len(cleaned_dataset['validation']):,} validation samples")
    print(f"Processed {len(cleaned_dataset['test']):,} test samples")
    print(f"TOTAL: {total_size:,} samples")
    
    # Label distribution
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION:")
    print("=" * 60)
    
    for split_name in ['train', 'validation', 'test']:
        labels = [ex['target'] for ex in cleaned_dataset[split_name]]
        safe = labels.count(0)
        vuln = labels.count(1)
        total = len(labels)
        
        print(f"\n{split_name.upper()}:")
        print(f"  Safe: {safe:,} ({safe/total*100:.1f}%)")
        print(f"  Vulnerable: {vuln:,} ({vuln/total*100:.1f}%)")
    
    return cleaned_dataset

def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description='Preprocess Devign dataset')
    parser.add_argument('--input', type=str, 
                       default='data/raw/devign_full_dataset',
                       help='Input dataset path')
    parser.add_argument('--output', type=str,
                       default='data/processed/devign_full_clean',
                       help='Output dataset path')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input dataset not found at {args.input}")
        print("\nPlease download the dataset first:")
        print("  python download_full_devign.py")
        return
    
    # Preprocess
    cleaned_dataset = prepare_full_dataset(
        args.input, 
        args.output,
        args.batch_size
    )
    
    print("\n✓ Dataset ready for training!")
    print(f"\nLoad with: load_from_disk('{args.output}')")
    print("\nNext step: Train the model")
    print("  python src/model/train.py")

if __name__ == "__main__":
    main()