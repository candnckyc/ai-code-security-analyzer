"""
Improved Dataset Download Script
Downloads vulnerability detection datasets with multiple fallback options
"""

import os
from datasets import load_dataset
import sys

def try_download_dataset(dataset_name, description):
    """Try to download a specific dataset"""
    print(f"\n{'='*60}")
    print(f"Attempting: {dataset_name}")
    print(f"Description: {description}")
    print('='*60)
    
    try:
        dataset = load_dataset(dataset_name)
        return dataset, dataset_name
    except Exception as e:
        print(f"✗ Failed: {str(e)[:100]}...")
        return None, None

def download_vulnerability_dataset():
    """Try multiple dataset sources"""
    print("=" * 60)
    print("Vulnerability Detection Dataset Downloader")
    print("=" * 60)
    print()
    
    # Create data directory
    os.makedirs("data/raw", exist_ok=True)
    
    # List of datasets to try (in order of preference)
    dataset_options = [
        # Option 1: Original Devign
        {
            "name": "Elfocrash/Devign",
            "description": "Original Devign dataset",
            "save_name": "devign"
        },
        # Option 2: Alternative Devign location
        {
            "name": "devign",
            "description": "Devign dataset (alternative)",
            "save_name": "devign"
        },
        # Option 3: Big-Vul dataset (larger, similar purpose)
        {
            "name": "ProgCorp/Big-Vul", 
            "description": "Big-Vul vulnerability dataset",
            "save_name": "big-vul"
        },
        # Option 4: Code vulnerability dataset
        {
            "name": "code-vulnerability/code-vulnerability-dataset",
            "description": "Code vulnerability dataset",
            "save_name": "code-vuln"
        },
        # Option 5: Juliet-style dataset
        {
            "name": "juliet-test-suite-c-cpp",
            "description": "Juliet test suite for C/C++",
            "save_name": "juliet"
        }
    ]
    
    dataset = None
    dataset_name = None
    save_name = None
    
    # Try each dataset option
    for option in dataset_options:
        print(f"\n[Trying {option['name']}...]")
        dataset, found_name = try_download_dataset(option['name'], option['description'])
        
        if dataset is not None:
            dataset_name = found_name
            save_name = option['save_name']
            print(f"\n✓ SUCCESS! Downloaded: {dataset_name}")
            break
    
    if dataset is None:
        print("\n" + "=" * 60)
        print("✗ ERROR: Could not download any dataset")
        print("=" * 60)
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Some datasets may require Hugging Face authentication")
        print("3. Try manually downloading from:")
        print("   https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection")
        print("\nAlternative: We can create a smaller sample dataset for testing")
        return False
    
    # Display statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    for split_name in dataset.keys():
        print(f"\n{split_name.upper()} split:")
        print(f"  Samples: {len(dataset[split_name]):,}")
        print(f"  Columns: {dataset[split_name].column_names}")
    
    # Show example
    print("\n" + "=" * 60)
    print("Example Sample")
    print("=" * 60)
    
    example = dataset['train'][0] if 'train' in dataset else list(dataset.values())[0][0]
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"{key}: {value[:200]}...")
        else:
            print(f"{key}: {value}")
    
    # Save locally
    print("\n" + "=" * 60)
    print("Saving Dataset")
    print("=" * 60)
    
    save_path = f"data/raw/{save_name}"
    dataset.save_to_disk(save_path)
    print(f"✓ Dataset saved to: {save_path}")
    
    # Create info file
    info_path = f"data/raw/{save_name}_info.txt"
    with open(info_path, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Downloaded successfully\n\n")
        for split_name in dataset.keys():
            f.write(f"{split_name}: {len(dataset[split_name])} samples\n")
    
    print(f"✓ Info saved to: {info_path}")
    
    print("\n" + "=" * 60)
    print("✓ SUCCESS!")
    print("=" * 60)
    print(f"\nDataset '{dataset_name}' is ready to use!")
    print("\nNext steps:")
    print("1. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("2. Or load in Python:")
    print(f"   from datasets import load_from_disk")
    print(f"   dataset = load_from_disk('{save_path}')")
    print()
    
    return True

def create_sample_dataset():
    """Create a small sample dataset for testing if downloads fail"""
    print("\n" + "=" * 60)
    print("Creating Sample Dataset")
    print("=" * 60)
    
    from datasets import Dataset, DatasetDict
    import pandas as pd
    
    # Sample vulnerable code
    vulnerable_samples = [
        {
            "func": """
            char* login(char* username, char* password) {
                char query[256];
                sprintf(query, "SELECT * FROM users WHERE username='%s' AND password='%s'", username, password);
                return execute_query(query);
            }
            """,
            "target": 1,
            "project": "sample",
            "commit_id": "sample1"
        },
        {
            "func": """
            void read_file(char* filename) {
                char buffer[100];
                FILE* f = fopen(filename, "r");
                gets(buffer);  // Buffer overflow vulnerability
                fclose(f);
            }
            """,
            "target": 1,
            "project": "sample",
            "commit_id": "sample2"
        },
        {
            "func": """
            def execute_command(user_input):
                os.system("cat " + user_input)  # Command injection
            """,
            "target": 1,
            "project": "sample",
            "commit_id": "sample3"
        }
    ]
    
    # Sample safe code
    safe_samples = [
        {
            "func": """
            char* login(char* username, char* password) {
                PreparedStatement* stmt = prepare("SELECT * FROM users WHERE username=? AND password=?");
                bind_string(stmt, 1, username);
                bind_string(stmt, 2, password);
                return execute(stmt);
            }
            """,
            "target": 0,
            "project": "sample",
            "commit_id": "sample4"
        },
        {
            "func": """
            void read_file(char* filename) {
                char buffer[100];
                FILE* f = fopen(filename, "r");
                fgets(buffer, sizeof(buffer), f);  // Safe with size limit
                fclose(f);
            }
            """,
            "target": 0,
            "project": "sample",
            "commit_id": "sample5"
        }
    ]
    
    # Combine samples
    all_samples = vulnerable_samples + safe_samples
    
    # Create dataset
    df = pd.DataFrame(all_samples)
    dataset = Dataset.from_pandas(df)
    
    # Create train/val/test splits
    train_test = dataset.train_test_split(test_size=0.4, seed=42)
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)
    
    dataset_dict = DatasetDict({
        'train': train_test['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    })
    
    # Save
    save_path = "data/raw/sample_dataset"
    dataset_dict.save_to_disk(save_path)
    
    print(f"✓ Created sample dataset with {len(all_samples)} examples")
    print(f"✓ Saved to: {save_path}")
    print("\nThis is a TINY dataset just for testing!")
    print("You should try to download a real dataset for actual training.")
    
    return True

if __name__ == "__main__":
    print("Starting dataset download...\n")
    
    success = download_vulnerability_dataset()
    
    if not success:
        print("\n" + "="*60)
        response = input("\nWould you like to create a small sample dataset for testing? (y/n): ")
        if response.lower() == 'y':
            create_sample_dataset()
        else:
            print("\nYou can try downloading manually later.")
            print("Check: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection")
    
    sys.exit(0 if success else 1)