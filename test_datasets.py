# test_dataset.py
from datasets import load_from_disk

# Load the sample dataset
dataset = load_from_disk("data/raw/sample_dataset")

print("✓ Dataset loaded successfully!")
print(f"\nSplits available: {list(dataset.keys())}")
print(f"Train samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")
print(f"Test samples: {len(dataset['test'])}")

print("\n" + "="*60)
print("Example vulnerable code:")
print("="*60)
print(dataset['train'][0]['func'])
print(f"Target (vulnerable): {dataset['train'][0]['target']}")