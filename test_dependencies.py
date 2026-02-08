#!/usr/bin/env python3
"""
Test script to check if all required dependencies are available
"""

def test_dependency(module_name, package_name=None):
    """Test if a dependency is available"""
    try:
        __import__(module_name)
        print(f"OK {package_name or module_name} is available")
        return True
    except ImportError as e:
        print(f"MISSING {package_name or module_name} is NOT available: {e}")
        return False

def main():
    """Test all required dependencies"""
    print("Testing Instagram AI Agent Dependencies")
    print("=" * 50)
    
    # Core dependencies
    dependencies = [
        ("json", "json"),
        ("logging", "logging"),
        ("datetime", "datetime"),
        ("os", "os"),
        ("pathlib", "pathlib"),
        ("random", "random"),
        ("typing", "typing"),
        ("dataclasses", "dataclasses"),
    ]
    
    # Optional AI dependencies
    ai_dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("PIL", "Pillow"),
        ("filestack", "Filestack"),
    ]
    
    # Required dependencies
    print("Required Dependencies:")
    all_required_ok = True
    for module, name in dependencies:
        if not test_dependency(module, name):
            all_required_ok = False
    
    print("\nOptional AI Dependencies:")
    for module, name in ai_dependencies:
        test_dependency(module, name)
    
    print("\n" + "=" * 50)
    if all_required_ok:
        print("OK All required dependencies are available!")
        print("OK Instagram AI Agent should work with template fallbacks")
    else:
        print("MISSING Some required dependencies are missing!")
        print("MISSING Instagram AI Agent may not work properly")

if __name__ == "__main__":
    main()