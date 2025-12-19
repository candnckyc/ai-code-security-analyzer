"""
Simple Streamlit UI for Code Security Analyzer
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Page config
st.set_page_config(
    page_title="AI Code Security Analyzer",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_model(model_path):
    """Load model (cached)"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def predict(code, tokenizer, model):
    """Make prediction"""
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        conf = probs[0][pred].item()
    
    return pred, conf, probs[0].tolist()

def main():
    st.title("🛡️ AI-Powered Code Security Analyzer")
    st.markdown("Detect security vulnerabilities in your code using AI")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This tool uses **CodeBERT** fine-tuned on vulnerability datasets 
        to detect security issues in source code.
        
        **Author:** Mahmut Can Dinçkuyucu  
        **Institution:** Mersin University  
        **Advisor:** Dr. Furkan Gözükara
        """)
        
        st.header("Detects:")
        st.markdown("""
        - SQL Injection
        - Buffer Overflow
        - Command Injection
        - And more...
        """)
    
    # Main content
    model_path = "models/finetuned/codebert-security/final"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.info("Please train the model first by running: `python src/model/train.py`")
        return
    
    # Load model
    with st.spinner("Loading model..."):
        tokenizer, model = load_model(model_path)
    
    st.success("✓ Model loaded!")
    
    # Input area
    st.subheader("📝 Enter Your Code")
    
    code_input = st.text_area(
        "Paste your code here:",
        height=300,
        placeholder="""char* login(char* username, char* password) {
    char query[256];
    sprintf(query, "SELECT * FROM users WHERE username='%s' AND password='%s'", username, password);
    return execute_query(query);
}"""
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analyze_btn = st.button("🔍 Analyze Code", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        st.rerun()
    
    if analyze_btn:
        if not code_input.strip():
            st.warning("⚠️ Please enter some code to analyze!")
        else:
            with st.spinner("Analyzing..."):
                prediction, confidence, probabilities = predict(code_input, tokenizer, model)
            
            # Results
            st.subheader("📊 Analysis Results")
            
            # Main result
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("### 🚨 VULNERABLE CODE DETECTED")
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                else:
                    st.success("### ✅ CODE APPEARS SAFE")
                    st.markdown(f"**Confidence:** {confidence:.1%}")
            
            with col2:
                # Probability breakdown
                st.markdown("**Probability Breakdown:**")
                st.progress(probabilities[1], text=f"Vulnerable: {probabilities[1]:.1%}")
                st.progress(probabilities[0], text=f"Safe: {probabilities[0]:.1%}")
            
            # Detailed analysis
            st.divider()
            
            if prediction == 1:
                st.subheader("⚠️ Potential Issues")
                st.markdown("""
                This code may contain security vulnerabilities. Common issues to check:
                
                - **SQL Injection:** Using string concatenation for SQL queries
                - **Buffer Overflow:** Using unsafe functions like `gets()`, `strcpy()`
                - **Command Injection:** Passing user input to system commands
                - **Format String:** Improper use of printf-style functions
                
                **Recommendation:** Review the code carefully and use secure coding practices.
                """)
            else:
                st.subheader("✅ Security Analysis")
                st.markdown("""
                The code appears to follow secure coding practices. However:
                
                - This is an AI prediction and not a guarantee
                - Always conduct thorough security reviews
                - Use additional security tools
                - Follow security best practices
                """)
    
    # Example codes
    st.divider()
    st.subheader("💡 Try These Examples")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Vulnerable Example (SQL Injection):**")
        vulnerable_code = '''char* login(char* user, char* pass) {
    char query[256];
    sprintf(query, "SELECT * FROM users WHERE user='%s'", user);
    return execute_query(query);
}'''
        st.code(vulnerable_code, language='c')
    
    with col2:
        st.markdown("**Safe Example (Parameterized Query):**")
        safe_code = '''char* login(char* user, char* pass) {
    PreparedStatement* stmt = prepare("SELECT * FROM users WHERE user=?");
    bind_string(stmt, 1, user);
    return execute(stmt);
}'''
        st.code(safe_code, language='c')

if __name__ == "__main__":
    main()