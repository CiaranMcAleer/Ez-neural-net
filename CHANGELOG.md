# Changelog

## [1.0.0] - 18/06/2025

### Added
- Initial release of Ez Vision
- `ImageClassifier` class for easy image classification
- `quick_classify()` function for one-line image classification
- Transfer learning using EfficientNetB0 for high accuracy
- Automatic model saving and loading with `.keras` format
- Auto-generated prediction scripts for saved models
- Comprehensive error handling and validation
- Support for common image formats (JPEG, PNG, etc.)
- Automatic image preprocessing and augmentation
- Interactive help system with `help()` function
- Complete example scripts in `examples/` directory
- Legacy neural network utilities (`nn_utils.py`) for general ML tasks

### Features
- **Beginner-friendly**: No ML experience required
- **Fast training**: Uses pre-trained models (transfer learning)
- **Flexible**: Works with any number of image categories
- **Robust**: Comprehensive test suite with 17 tests
- **Production-ready**: Proper model saving/loading and deployment scripts

### Examples Included
- Basic classification walkthrough
- Quick mode for rapid prototyping
- Model saving and loading demonstration
- Interactive demo script

### Technical Details
- Built on TensorFlow/Keras
- Supports Python 3.8+
- Automatic data validation and preprocessing
- GPU acceleration support (when available)
- Time-limited training with early stopping
- Detailed logging and progress tracking
