"""
Ez Vision - Computer Vision Made Simple

A beginner-friendly computer vision library that makes training 
image recognition models as easy as a few lines of code.

Example:
    from ez_vision import ImageClassifier, quick_classify
    
    # Method 1: Full control
    classifier = ImageClassifier()
    classifier.train('my_photos/')
    result = classifier.predict('test_image.jpg')
    
    # Method 2: One-liner
    result = quick_classify('my_photos/', 'test_image.jpg')
"""

__version__ = "1.0.0"
__author__ = "Ez Vision Team"
__email__ = "contact@ez-vision.ai"

# Import main classes and functions
from .ez_vision import ImageClassifier, ObjectFinder, quick_classify, help

# Make these available at package level
__all__ = [
    "ImageClassifier",
    "ObjectFinder", 
    "quick_classify",
    "help",
    "__version__"
]
