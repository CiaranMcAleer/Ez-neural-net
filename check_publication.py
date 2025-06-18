#!/usr/bin/env python3
"""
Publication Readiness Check for Ez Vision

This script validates that the package is ready for publication.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_file_exists(filename):
    """Check if a required file exists"""
    if os.path.exists(filename):
        print(f"✅ {filename} exists")
        return True
    else:
        print(f"❌ {filename} missing")
        return False

def check_package_structure():
    """Check that the package has proper structure"""
    print("\n📦 Checking package structure...")
    
    required_files = [
        "README.md",
        "LICENSE", 
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
        "MANIFEST.in",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "__init__.py",
        "ez_vision.py",
        "nn_utils.py",
    ]
    
    all_exist = True
    for file in required_files:
        if not check_file_exists(file):
            all_exist = False
    
    return all_exist

def check_examples():
    """Check that example files exist"""
    print("\n📚 Checking examples...")
    
    examples_dir = "examples"
    if not os.path.exists(examples_dir):
        print(f"❌ {examples_dir} directory missing")
        return False
    
    required_examples = [
        "basic_classification.py",
        "quick_mode.py", 
        "load_saved_model.py"
    ]
    
    all_exist = True
    for example in required_examples:
        example_path = os.path.join(examples_dir, example)
        if not check_file_exists(example_path):
            all_exist = False
    
    return all_exist

def check_imports():
    """Check that the package imports correctly"""
    print("\n🔍 Checking imports...")
    
    try:
        # Test importing the main module
        sys.path.insert(0, '.')
        import ez_vision
        print("✅ ez_vision imports successfully")
        
        # Test importing main classes
        from ez_vision import ImageClassifier, quick_classify, help
        print("✅ Main classes import successfully")
        
        # Test help function
        import io
        from contextlib import redirect_stdout
        output = io.StringIO()
        with redirect_stdout(output):
            help()
        help_text = output.getvalue()
        
        if len(help_text) > 100:
            print("✅ Help function works")
        else:
            print("❌ Help function may have issues")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def check_version_consistency():
    """Check that version numbers are consistent"""
    print("\n🔢 Checking version consistency...")
    
    # Check __init__.py version
    try:
        import re
        with open('__init__.py', 'r') as f:
            init_content = f.read()
        
        version_match = re.search(r'__version__ = ["\']([^"\']+)["\']', init_content)
        if version_match:
            init_version = version_match.group(1)
            print(f"✅ __init__.py version: {init_version}")
        else:
            print("❌ Version not found in __init__.py")
            return False
        
        # Check setup.py version
        with open('setup.py', 'r') as f:
            setup_content = f.read()
        
        setup_version_match = re.search(r'version=["\']([^"\']+)["\']', setup_content)
        if setup_version_match:
            setup_version = setup_version_match.group(1)
            print(f"✅ setup.py version: {setup_version}")
        else:
            print("❌ Version not found in setup.py")
            return False
        
        # Check pyproject.toml version
        with open('pyproject.toml', 'r') as f:
            toml_content = f.read()
        
        toml_version_match = re.search(r'version = ["\']([^"\']+)["\']', toml_content)
        if toml_version_match:
            toml_version = toml_version_match.group(1)
            print(f"✅ pyproject.toml version: {toml_version}")
        else:
            print("❌ Version not found in pyproject.toml")
            return False
        
        # Check if all versions match
        if init_version == setup_version == toml_version:
            print(f"✅ All versions consistent: {init_version}")
            return True
        else:
            print("❌ Version mismatch between files")
            return False
            
    except Exception as e:
        print(f"❌ Error checking versions: {e}")
        return False

def run_publication_check():
    """Run all publication readiness checks"""
    print("🚀 Ez Vision Publication Readiness Check")
    print("=" * 50)
    
    checks = [
        ("Package Structure", check_package_structure),
        ("Examples", check_examples), 
        ("Imports", check_imports),
        ("Version Consistency", check_version_consistency),
    ]
    
    all_passed = True
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            results.append((name, False))
            all_passed = False
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    if all_passed:
        print("\n🎉 Package is ready for publication!")
        print("\n📝 Next steps:")
        print("1. Run tests: python tests.py")
        print("2. Build package: python -m build")
        print("3. Test install: pip install dist/ez_vision-*.whl")
        print("4. Upload to PyPI: twine upload dist/*")
        return True
    else:
        print("\n❌ Package needs fixes before publication")
        return False

if __name__ == "__main__":
    success = run_publication_check()
    sys.exit(0 if success else 1)
