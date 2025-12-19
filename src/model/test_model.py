"""
Inference Script
Test the trained model on new code samples
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class VulnerabilityDetector:
    def __init__(self, model_path):
        """Initialize detector with trained model"""
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        print("✓ Model loaded!")
        
    def predict(self, code):
        """
        Predict if code is vulnerable
        
        Args:
            code: Source code string
            
        Returns:
            dict with prediction, confidence, and probabilities
        """
        # Tokenize
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            'is_vulnerable': bool(prediction),
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'safe': float(probabilities[0][0]),
                'vulnerable': float(probabilities[0][1])
            }
        }
    
    def predict_batch(self, codes):
        """Predict for multiple code samples"""
        results = []
        for code in codes:
            results.append(self.predict(code))
        return results

def test_model():
    """Test the model with example code"""
    
    # Load model
    model_path = "models/finetuned/codebert-security/final"
    detector = VulnerabilityDetector(model_path)
    
    # Test cases
    test_cases = [
        {
            'name': 'SQL Injection (Vulnerable)',
            'code': '''
            char* login(char* username, char* password) {
                char query[256];
                sprintf(query, "SELECT * FROM users WHERE username='%s' AND password='%s'", 
                        username, password);
                return execute_query(query);
            }
            '''
        },
        {
            'name': 'Buffer Overflow (Vulnerable)',
            'code': '''
            void read_input() {
                char buffer[10];
                gets(buffer);  // Dangerous!
                printf("%s", buffer);
            }
            '''
        },
        {
            'name': 'Safe Parameterized Query',
            'code': '''
            char* login(char* username, char* password) {
                PreparedStatement* stmt = prepare("SELECT * FROM users WHERE username=? AND password=?");
                bind_string(stmt, 1, username);
                bind_string(stmt, 2, password);
                return execute(stmt);
            }
            '''
        },
        {
            'name': 'Safe Buffer Handling',
            'code': '''
            void read_input() {
                char buffer[10];
                fgets(buffer, sizeof(buffer), stdin);
                printf("%s", buffer);
            }
            '''
        }
    ]
    
    print("\n" + "=" * 60)
    print("TESTING MODEL PREDICTIONS")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test['name']}")
        print("-" * 60)
        print("Code:")
        print(test['code'][:150] + "...")
        print()
        
        result = detector.predict(test['code'])
        
        print(f"Prediction: {'VULNERABLE' if result['is_vulnerable'] else 'SAFE'}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities:")
        print(f"  Safe: {result['probabilities']['safe']:.2%}")
        print(f"  Vulnerable: {result['probabilities']['vulnerable']:.2%}")
        
        # Check if prediction is correct (based on test name)
        expected_vulnerable = 'Vulnerable' in test['name']
        is_correct = result['is_vulnerable'] == expected_vulnerable
        print(f"\n{'✓ CORRECT' if is_correct else '✗ INCORRECT'} prediction!")

def main():
    """Main function"""
    import os
    
    model_path = "models/finetuned/codebert-security/final"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please run train.py first!")
        return
    
    test_model()

if __name__ == "__main__":
    main()