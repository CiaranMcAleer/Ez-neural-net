"""
Test script for Ez Vision
Run this to make sure everything is working correctly!
"""

import os
import sys

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"  ✅ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"  ❌ TensorFlow import failed: {e}")
        return False
        
    try:
        import numpy as np
        print(f"  ✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ❌ NumPy import failed: {e}")
        return False
        
    try:
        import matplotlib
        print(f"  ✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ❌ Matplotlib import failed: {e}")
        return False
        
    try:
        import sklearn
        print(f"  ✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"  ❌ Scikit-learn import failed: {e}")
        return False
        
    try:
        from PIL import Image
        print(f"  ✅ Pillow (PIL)")
    except ImportError as e:
        print(f"  ❌ Pillow import failed: {e}")
        return False
        
    return True

def test_ez_vision():
    """Test that Ez Vision modules load correctly"""
    print("\n🎨 Testing Ez Vision...")
    
    try:
        from ez_vision import ImageClassifier, help, quick_classify
        print("  ✅ Ez Vision imports successful")
        
        # Test help function
        print("  ✅ Help function available")
        
        # Test classifier creation
        classifier = ImageClassifier()
        print("  ✅ ImageClassifier creation successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ez Vision test failed: {e}")
        return False

def test_legacy_nn_utils():
    """Test that the original neural network utilities still work"""
    print("\n🧠 Testing legacy neural network utilities...")
    
    try:
        from nn_utils import train_neural_network, validate_input_data
        print("  ✅ Legacy nn_utils imports successful")
        return True
    except Exception as e:
        print(f"  ❌ Legacy nn_utils test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Ez Vision Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        
    # Test Ez Vision
    if not test_ez_vision():
        all_passed = False
        
    # Test legacy utilities
    if not test_legacy_nn_utils():
        all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All tests passed! Ez Vision is ready to use.")
        print("\nNext steps:")
        print("1. Check out the examples/ folder")
        print("2. Read SETUP.md for getting started")
        print("3. Run: python demo.py for help")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nTry running: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
