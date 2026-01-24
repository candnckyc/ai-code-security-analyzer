"""
Test Trained Model - FINAL VERSION
Test the model trained on full Devign dataset
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path):
    """Load trained model and tokenizer"""
    print("=" * 60)
    print("LOADING TRAINED MODEL")
    print("=" * 60)
    
    print(f"\nModel path: {model_path}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Set to eval mode
    model.eval()
    
    print(f"✓ Model loaded: {model.num_parameters():,} parameters")
    print("✓ Ready for inference!")
    
    return tokenizer, model

def predict(code, tokenizer, model, max_length=512):
    """Predict if code is vulnerable"""
    
    # Tokenize
    inputs = tokenizer(
        code,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()
    
    return prediction, confidence, probs[0].tolist()

def test_model(model_path):
    """Test model with examples"""
    
    # Load model
    tokenizer, model = load_model(model_path)
    
    # Test examples
    test_cases = [
        {
            'name': 'SQL Injection (Vulnerable)',
            'code': '''char* login(char* username, char* password) {
    char query[256];
    sprintf(query, "SELECT * FROM users WHERE username='%s' AND password='%s'", 
            username, password);
    return execute_query(query);
}''',
            'expected': 1
        },
        {
            'name': 'Buffer Overflow (Vulnerable)',
            'code': '''void read_input() {
    char buffer[10];
    gets(buffer);
    printf("%s", buffer);
}''',
            'expected': 1
        },
        {
            'name': 'Parameterized Query (Safe)',
            'code': '''char* login(char* username, char* password) {
    PreparedStatement* stmt = prepare("SELECT * FROM users WHERE username=? AND password=?");
    bind_string(stmt, 1, username);
    bind_string(stmt, 2, password);
    return execute(stmt);
}''',
            'expected': 0
        },
        {
            'name': 'Safe Buffer (Safe)',
            'code': '''void read_input() {
    char buffer[256];
    fgets(buffer, sizeof(buffer), stdin);
    buffer[strcspn(buffer, "\\n")] = 0;
    printf("Input: %s\\n", buffer);
}''',
            'expected': 0
        },
        {
            'name': 'Command Injection (Vulnerable)',
            'code': '''void execute_command(char* user_input) {
    char command[256];
    sprintf(command, "ping %s", user_input);
    system(command);
}''',
            'expected': 1
        }
    ]
    
    print("\n" + "=" * 60)
    print("TESTING MODEL")
    print("=" * 60)
    
    correct = 0
    total = len(test_cases)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'='*60}")
        
        # Predict
        prediction, confidence, probs = predict(test['code'], tokenizer, model)
        
        # Label
        label = "VULNERABLE" if prediction == 1 else "SAFE"
        expected_label = "VULNERABLE" if test['expected'] == 1 else "SAFE"
        
        # Correct?
        is_correct = (prediction == test['expected'])
        if is_correct:
            correct += 1
            status = "✓ CORRECT"
        else:
            status = "✗ WRONG"
        
        # Display
        print(f"\nPrediction: {label}")
        print(f"Expected:   {expected_label}")
        print(f"Confidence: {confidence*100:.1f}%")
        print(f"Probabilities: Safe={probs[0]*100:.1f}%, Vulnerable={probs[1]*100:.1f}%")
        print(f"\nResult: {status}")
        
        # Show code snippet
        print(f"\nCode snippet:")
        print("-" * 60)
        print(test['code'][:200] + "..." if len(test['code']) > 200 else test['code'])
        print("-" * 60)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/total*100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    # Model path
    model_path = "models/finetuned/codebert-security-full/final"
    
    # Test
    test_model(model_path)