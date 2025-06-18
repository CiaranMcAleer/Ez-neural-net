
# Ez Vision - Computer Vision Made Simple! 🎨

**Train powerful image recognition models in just a few lines of code - no AI knowledge required!**

Ez Vision makes computer vision accessible to everyone. Whether you want to:
- Recognize different types of photos (cats vs dogs vs birds)
- Sort your image collection automatically  
- Build smart apps that understand images
- Learn about AI in a fun, hands-on way

You can do it with just 10-20 lines of Python!

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Super Simple Example
```python
from ez_vision import ImageClassifier

# Create a classifier
classifier = ImageClassifier()

# Train it on your photos (organize in folders by category)
classifier.train('my_photos/')

# Use it to recognize new images
result = classifier.predict('new_photo.jpg')
print(f"This looks like: {result}")
```

### Even Simpler - One Line Mode!
```python
from ez_vision import quick_classify

result = quick_classify('my_photos/', 'test_image.jpg')
print(f"This is a: {result}")
```

## 📁 How to Organize Your Images

Just put your images in folders by category:

```
my_photos/
    cats/
        cat1.jpg
        cat2.jpg
        cat3.jpg
    dogs/
        dog1.jpg
        dog2.jpg
    birds/
        bird1.jpg
        bird2.jpg
```

That's it! Ez Vision will automatically learn the differences.

## 🎯 What Can You Build?

- **Photo organizer**: Automatically sort vacation photos
- **Pet classifier**: Tell apart different dog breeds
- **Food recognition**: Identify different dishes
- **Plant identifier**: Recognize flowers and trees
- **Quality checker**: Sort good vs bad product photos
- **Art classifier**: Categorize different art styles

## 📚 More Examples

Check out the `examples/` folder for:
- `basic_classification.py` - Full walkthrough with explanations
- `quick_mode.py` - Super fast one-liner approach  
- `load_saved_model.py` - How to save and reuse your models

## 💡 Tips for Better Results

- Use at least 20 images per category
- More categories = more training time needed
- JPEG, PNG, and most image formats work
- The AI automatically handles image resizing

## 🛠️ Legacy Neural Network Utils

The original `nn_utils.py` is still available for general neural network tasks, but Ez Vision is optimized specifically for images and much easier to use!

---

**No PhD required - just point, click, and let the AI learn!** 🧠✨

