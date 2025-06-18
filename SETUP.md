# Quick Setup Guide for Ez Vision

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the installation:**
   ```bash
   python demo.py
   ```

## Getting Started in 3 Steps

### Step 1: Organize Your Images
Create folders with your image categories:
```
my_training_data/
    cats/
        cat1.jpg
        cat2.jpg
        cat3.jpg
    dogs/
        dog1.jpg
        dog2.jpg
        dog3.jpg
```

### Step 2: Train Your Model
```python
from ez_vision import ImageClassifier

classifier = ImageClassifier()
classifier.train('my_training_data/', epochs=15)
classifier.save('my_model/')
```

### Step 3: Use Your Model
```python
# Load and use your saved model
classifier = ImageClassifier()
classifier.load('my_model/')

result = classifier.predict('new_photo.jpg')
print(f"This looks like: {result}")
```

## Super Quick Mode
For the fastest experience:
```python
from ez_vision import quick_classify

result = quick_classify('my_training_data/', 'test_photo.jpg')
print(f"Prediction: {result}")
```

## Example Projects
- **Pet classifier**: Recognize cat vs dog vs bird
- **Food recognition**: Identify different dishes
- **Plant identifier**: Recognize different flowers
- **Quality checker**: Sort good vs bad product photos

## Need Help?
```python
from ez_vision import help
help()  # Shows detailed usage guide
```

## Tips for Better Results
- Use at least 20 images per category
- Make sure images are clear and well-lit
- Include variety in your training images
- More epochs = better accuracy (but slower training)
- JPEG, PNG, and most image formats work fine
