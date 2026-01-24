"""
Fix Dataset Labels - Convert Boolean to Integer
Quick fix for training error
"""

from datasets import load_from_disk

def fix_labels(dataset_path, output_path):
    """Convert boolean labels to integers"""
    
    print("=" * 60)
    print("FIXING DATASET LABELS")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Check current type
    print("\nCurrent label types:")
    for split in ['train', 'validation', 'test']:
        first_label = dataset[split][0]['target']
        print(f"  {split}: {type(first_label)} = {first_label}")
    
    # Convert boolean to int
    print("\nConverting boolean → integer...")
    
    def convert_label(example):
        # Convert True/False to 1/0
        example['target'] = int(example['target'])
        return example
    
    # Apply to all splits
    for split in ['train', 'validation', 'test']:
        print(f"  Processing {split}...")
        dataset[split] = dataset[split].map(convert_label)
    
    # Verify
    print("\nNew label types:")
    for split in ['train', 'validation', 'test']:
        first_label = dataset[split][0]['target']
        print(f"  {split}: {type(first_label)} = {first_label}")
    
    # Check distribution
    print("\nLabel distribution:")
    for split in ['train', 'validation', 'test']:
        labels = [ex['target'] for ex in dataset[split]]
        print(f"  {split}: 0={labels.count(0):,}, 1={labels.count(1):,}")
    
    # Save
    print(f"\nSaving to: {output_path}")
    dataset.save_to_disk(output_path)
    
    print("\n" + "=" * 60)
    print("✓ LABELS FIXED!")
    print("=" * 60)
    print("\nNow you can train:")
    print("  python src/model/train_full.py")

if __name__ == "__main__":
    input_path = "data/raw/devign_full_dataset"
    output_path = "data/raw/devign_full_dataset_fixed"
    
    fix_labels(input_path, output_path)
    
    # Update symlink
    import os
    if os.path.exists(input_path + "_backup"):
        os.remove(input_path + "_backup")
    
    print("\nRenaming:")
    print(f"  {input_path} → {input_path}_backup")
    print(f"  {output_path} → {input_path}")
    
    try:
        os.rename(input_path, input_path + "_backup")
        os.rename(output_path, input_path)
        print("\n✓ Dataset replaced!")
    except Exception as e:
        print(f"\n⚠️  Manual rename needed: {e}")
        print("\nManually:")
        print(f"1. Rename: {input_path} → {input_path}_backup")
        print(f"2. Rename: {output_path} → {input_path}")