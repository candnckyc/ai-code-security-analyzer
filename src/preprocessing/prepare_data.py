"""
Data Preparation Script
Loads, cleans, and prepares dataset for training
"""

import os
import sys
from datasets import load_from_disk, DatasetDict
from tqdm import tqdm

# Add parent directory to path to import parser
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocessing.parser import clean_code

def preprocess_example(example):
    """Preprocess a single example"""
    example['func'] = clean_code(example['func'])
    return example

def prepare_dataset(input_path, output_path):
    """
    Load raw dataset, clean it, and save processed version
    
    Args:
        input_path: Path to raw dataset
        output_path: Path to save processed dataset
    """
    print("=" * 60)
    print("Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Step 1: Load raw dataset
    print("\n[1/4] Loading raw dataset...")
    dataset = load_from_disk(input_path)
    print(f"✓ Loaded {len(dataset['train'])} train samples")
    print(f"✓ Loaded {len(dataset['validation'])} validation samples")
    print(f"✓ Loaded {len(dataset['test'])} test samples")
    
    # Step 2: Show example before processing
    print("\n[2/4] Example BEFORE preprocessing:")
    print("-" * 60)
    example = dataset['train'][0]['func'][:200]
    print(example + "...")
    
    # Step 3: Clean all splits
    print("\n[3/4] Cleaning code samples...")
    
    print("  Cleaning train split...")
    train_clean = dataset['train'].map(
        preprocess_example,
        desc="Cleaning train"
    )
    
    print("  Cleaning validation split...")
    val_clean = dataset['validation'].map(
        preprocess_example,
        desc="Cleaning validation"
    )
    
    print("  Cleaning test split...")
    test_clean = dataset['test'].map(
        preprocess_example,
        desc="Cleaning test"
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
    print(f"Processed {len(cleaned_dataset['train'])} train samples")
    print(f"Processed {len(cleaned_dataset['validation'])} validation samples")
    print(f"Processed {len(cleaned_dataset['test'])} test samples")
    
    return cleaned_dataset

def main():
    """Main preprocessing function"""
    # Paths
    raw_path = "data/raw/sample_dataset"
    processed_path = "data/processed/sample_dataset_clean"
    
    # Check if raw dataset exists
    if not os.path.exists(raw_path):
        print(f"ERROR: Raw dataset not found at {raw_path}")
        print("Please run download_dataset.py first!")
        return
    
    # Preprocess
    cleaned_dataset = prepare_dataset(raw_path, processed_path)
    
    print("\n✓ Dataset ready for training!")
    print(f"Load with: load_from_disk('{processed_path}')")

if __name__ == "__main__":
    main()