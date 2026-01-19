"""
Fix Dataset Distribution
3 vulnerable, 2 safe olacak şekilde düzelt
"""

from datasets import load_from_disk, Dataset, DatasetDict

# Mevcut veriyi yükle
dataset = load_from_disk('data/raw/sample_dataset')

print("MEVCUT DURUM:")
print("="*60)
for split_name in ['train', 'validation', 'test']:
    labels = [ex['target'] for ex in dataset[split_name]]
    print(f"{split_name}: {labels.count(0)} safe, {labels.count(1)} vulnerable")

# YENİ VERİ: 3 vulnerable, 2 safe
# Train: 2 vulnerable, 1 safe
# Validation: 1 safe
# Test: 1 vulnerable

# TRAIN SET (3 örnek: 2 vulnerable, 1 safe)
train_examples = [
    # Vulnerable 1: SQL Injection
    {
        'func': '''char* login(char* username, char* password) {
    char query[256];
    sprintf(query, "SELECT * FROM users WHERE username='%s' AND password='%s'", 
            username, password);
    return execute_query(query);
}''',
        'target': 1
    },
    # Vulnerable 2: Buffer Overflow
    {
        'func': '''void read_input() {
    char buffer[10];
    gets(buffer);
    printf("%s", buffer);
}''',
        'target': 1
    },
    # Safe 1: Parameterized Query
    {
        'func': '''char* login(char* username, char* password) {
    PreparedStatement* stmt = prepare("SELECT * FROM users WHERE username=? AND password=?");
    bind_string(stmt, 1, username);
    bind_string(stmt, 2, password);
    return execute(stmt);
}''',
        'target': 0
    }
]

# VALIDATION SET (1 örnek: 1 safe)
validation_examples = [
    # Safe: fgets ile güvenli buffer
    {
        'func': '''void read_input() {
    char buffer[10];
    fgets(buffer, sizeof(buffer), stdin);
    printf("%s", buffer);
}''',
        'target': 0
    }
]

# TEST SET (1 örnek: 1 vulnerable)
test_examples = [
    # Vulnerable: Command Injection
    {
        'func': '''void execute_command(char* user_input) {
    char command[256];
    sprintf(command, "ping %s", user_input);
    system(command);
}''',
        'target': 1
    }
]

# Dataset'leri oluştur
train_dataset = Dataset.from_list(train_examples)
val_dataset = Dataset.from_list(validation_examples)
test_dataset = Dataset.from_list(test_examples)

# DatasetDict oluştur
new_dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

# Kaydet
new_dataset.save_to_disk('data/raw/sample_dataset_fixed')

print("\n" + "="*60)
print("YENİ DURUM:")
print("="*60)
for split_name in ['train', 'validation', 'test']:
    labels = [ex['target'] for ex in new_dataset[split_name]]
    safe = labels.count(0)
    vuln = labels.count(1)
    print(f"{split_name}: {safe} safe, {vuln} vulnerable")

print("\n" + "="*60)
print("TOPLAM:")
all_labels = []
for split_name in ['train', 'validation', 'test']:
    all_labels.extend([ex['target'] for ex in new_dataset[split_name]])
print(f"Safe: {all_labels.count(0)}")
print(f"Vulnerable: {all_labels.count(1)}")
print(f"Total: {len(all_labels)}")

print("\n✓ Düzeltilmiş dataset kaydedildi: data/raw/sample_dataset_fixed")
print("\nŞimdi şunu yap:")
print("1. Eski dataset'i yedekle: rename sample_dataset → sample_dataset_old")
print("2. Yeni dataset'i kullan: rename sample_dataset_fixed → sample_dataset")
print("3. Preprocessing'i tekrar çalıştır: python src/preprocessing/prepare_data.py")
print("4. Model'i tekrar eğit: python src/model/train.py")