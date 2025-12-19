"""
Code Parser Module
Cleans and normalizes source code for model input
"""

import re

def remove_comments(code):
    """Remove single-line and multi-line comments"""
    # Remove single-line comments (// and #)
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    code = re.sub(r'#.*?$', '', code, flags=re.MULTILINE)
    
    # Remove multi-line comments (/* */ and ''' ''')
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    
    return code

def normalize_whitespace(code):
    """Normalize whitespace and newlines"""
    # Replace multiple spaces with single space
    code = re.sub(r' +', ' ', code)
    
    # Replace multiple newlines with single newline
    code = re.sub(r'\n+', '\n', code)
    
    # Remove leading/trailing whitespace
    code = code.strip()
    
    return code

def clean_code(code):
    """Complete cleaning pipeline"""
    if not code or not isinstance(code, str):
        return ""
    
    # Step 1: Remove comments
    code = remove_comments(code)
    
    # Step 2: Normalize whitespace
    code = normalize_whitespace(code)
    
    return code

def test_parser():
    """Test the parser with sample code"""
    test_code = """
    // This is a comment
    char* login(char* user, char* pass) {
        /* Multi-line
           comment here */
        sprintf(query, "SELECT * FROM users WHERE user='%s'", user);
        return query;  # Another comment
    }
    """
    
    print("ORIGINAL CODE:")
    print("=" * 60)
    print(test_code)
    
    cleaned = clean_code(test_code)
    
    print("\nCLEANED CODE:")
    print("=" * 60)
    print(cleaned)
    print("\nLength before:", len(test_code))
    print("Length after:", len(cleaned))

if __name__ == "__main__":
    test_parser()