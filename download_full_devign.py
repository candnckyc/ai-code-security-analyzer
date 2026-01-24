"""
Download Devign Dataset from HuggingFace
Alternative method - more reliable!
"""

from datasets import load_dataset
import os

def download_from_huggingface():
    """
    Download Devign dataset from HuggingFace
    This is more reliable than GitHub direct download
    """
    print("=" * 60)
    print("DOWNLOADING DEVIGN FROM HUGGINGFACE")
    print("=" * 60)
    
    print("\nThis may take 5-10 minutes...")
    print("Total size: ~90MB")
    
    try:
        # Load from HuggingFace
        print("\n[1/2] Downloading from HuggingFace datasets...")
        dataset = load_dataset("code_x_glue_cc_defect_detection")
        
        print("\n✓ Download complete!")
        
        # Show statistics
        print("\n[2/2] Dataset Statistics:")
        print("=" * 60)
        
        for split in ['train', 'validation', 'test']:
            labels = [ex['target'] for ex in dataset[split]]
            safe = labels.count(0)
            vuln = labels.count(1)
            total = len(labels)
            
            print(f"\n{split.upper()} SET:")
            print(f"  Total: {total:,}")
            print(f"  Safe: {safe:,} ({safe/total*100:.1f}%)")
            print(f"  Vulnerable: {vuln:,} ({vuln/total*100:.1f}%)")
        
        # Save to disk
        save_path = "data/raw/devign_full_dataset"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"\nSaving to: {save_path}")
        dataset.save_to_disk(save_path)
        
        print("\n" + "=" * 60)
        print("✓ DOWNLOAD COMPLETE!")
        print("=" * 60)
        
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print("\n1. Preprocess the dataset:")
        print("   python src/preprocessing/prepare_data_full.py")
        print("\n2. Train the model:")
        print("   python src/model/train_full.py")
        
        return save_path
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nIf this fails, you may need to:")
        print("1. Update datasets library: pip install -U datasets")
        print("2. Or download manually from GitHub")
        return None

if __name__ == "__main__":
    download_from_huggingface()