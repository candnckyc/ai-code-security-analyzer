"""
Streamlit Web UI - FINAL VERSION
Using the trained model on full Devign dataset
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page config
st.set_page_config(
    page_title="AI Code Security Analyzer",
    page_icon="🔒",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load model and tokenizer (cached)"""
    model_path = "models/finetuned/codebert-security-full/final"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)

def predict_vulnerability(code, tokenizer, model, max_length=512):
    """Predict if code contains vulnerability"""
    
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

# Main UI
st.title("🔒 AI-Powered Code Security Analyzer")
st.markdown("### Detect security vulnerabilities in your code using AI")

# Load model
with st.spinner("Loading AI model..."):
    tokenizer, model, error = load_model()

if error:
    st.error(f"❌ Error loading model: {error}")
    st.info("Make sure the trained model is in: models/finetuned/codebert-security-full/final/")
    st.stop()

st.success("✅ Model loaded successfully!")

# Model info
with st.expander("ℹ️ Model Information"):
    st.markdown("""
    **Model:** CodeBERT (microsoft/codebert-base)
    
    **Training Dataset:** Devign (27,318 samples)
    - Train: 21,854 samples
    - Validation: 2,732 samples
    - Test: 2,732 samples
    
    **Performance:**
    - Accuracy: 66.1%
    - Precision: 67.9%
    - F1 Score: 55.8%
    
    **Detected Vulnerabilities:**
    - SQL Injection
    - Buffer Overflow
    - Command Injection
    - Use After Free
    - NULL Pointer Dereference
    - And more...
    """)

# Input section
st.markdown("---")
st.markdown("### 📝 Enter Your Code")

# Example codes
example_codes = {
    "SQL Injection (Vulnerable)": '''char* login(char* username, char* password) {
    char query[256];
    sprintf(query, "SELECT * FROM users WHERE username='%s' AND password='%s'", 
            username, password);
    return execute_query(query);
}''',
    "Buffer Overflow (Vulnerable)": '''void read_input() {
    char buffer[10];
    gets(buffer);
    printf("%s", buffer);
}''',
    "Parameterized Query (Safe)": '''char* login(char* username, char* password) {
    PreparedStatement* stmt = prepare("SELECT * FROM users WHERE username=? AND password=?");
    bind_string(stmt, 1, username);
    bind_string(stmt, 2, password);
    return execute(stmt);
}''',
    "Safe Buffer (Safe)": '''void read_input() {
    char buffer[256];
    fgets(buffer, sizeof(buffer), stdin);
    buffer[strcspn(buffer, "\\n")] = 0;
    printf("Input: %s\\n", buffer);
}'''
}

# Example selector
col1, col2 = st.columns([3, 1])
with col1:
    selected_example = st.selectbox(
        "Load Example:",
        ["Custom Code"] + list(example_codes.keys())
    )

if selected_example == "Custom Code":
    default_code = "// Enter your code here"
else:
    default_code = example_codes[selected_example]

# Code input
code_input = st.text_area(
    "Code to Analyze:",
    value=default_code,
    height=300,
    help="Enter C/C++/Python code to analyze for security vulnerabilities"
)

# Analyze button
if st.button("🔍 Analyze Code", type="primary"):
    if not code_input or code_input.strip() == "// Enter your code here":
        st.warning("⚠️ Please enter some code to analyze!")
    else:
        with st.spinner("Analyzing code..."):
            # Predict
            prediction, confidence, probs = predict_vulnerability(
                code_input, tokenizer, model
            )
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            # Result columns
            col1, col2, col3 = st.columns(3)
            
            # Prediction
            with col1:
                if prediction == 1:
                    st.error("🚨 **VULNERABLE CODE DETECTED**")
                    vulnerability_type = "HIGH RISK"
                else:
                    st.success("✅ **CODE APPEARS SAFE**")
                    vulnerability_type = "LOW RISK"
            
            # Confidence
            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence*100:.1f}%",
                    delta=None
                )
            
            # Risk Level
            with col3:
                st.metric(
                    "Risk Level",
                    vulnerability_type
                )
            
            # Probability breakdown
            st.markdown("#### 📈 Probability Breakdown")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Safe Probability", f"{probs[0]*100:.1f}%")
            
            with col2:
                st.metric("Vulnerable Probability", f"{probs[1]*100:.1f}%")
            
            # Progress bars
            st.progress(probs[0], text=f"Safe: {probs[0]*100:.1f}%")
            st.progress(probs[1], text=f"Vulnerable: {probs[1]*100:.1f}%")
            
            # Recommendations
            st.markdown("---")
            st.markdown("### 💡 Recommendations")
            
            if prediction == 1:
                st.warning("""
                **⚠️ Security Issues Detected:**
                
                This code may contain security vulnerabilities. Common issues include:
                - SQL Injection attacks
                - Buffer overflow vulnerabilities
                - Command injection risks
                - Memory safety issues
                
                **Recommendations:**
                1. Review the code for input validation
                2. Use parameterized queries for database operations
                3. Implement proper bounds checking for buffers
                4. Sanitize user inputs
                5. Use secure coding practices
                6. Consider code review and penetration testing
                """)
            else:
                st.info("""
                **✅ No Critical Issues Found**
                
                The AI model did not detect major security vulnerabilities. However:
                - This is not a guarantee of complete security
                - Always follow secure coding best practices
                - Consider manual code review
                - Use additional security testing tools
                - Keep dependencies updated
                """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>AI-Powered Code Security Analyzer</strong></p>
    <p>Mersin University - Computer Engineering Graduation Project</p>
    <p>Powered by CodeBERT & PyTorch</p>
</div>
""", unsafe_allow_html=True)