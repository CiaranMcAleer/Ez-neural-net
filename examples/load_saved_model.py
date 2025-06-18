"""
Example 3: Loading and Using a Saved Model

This shows how to load a previously trained model and use it.
Great for using your models in other projects!
"""

from ez_vision import ImageClassifier

# Create a new classifier
classifier = ImageClassifier()

# Load your previously saved model
print("Loading saved model...")
classifier.load('my_trained_model/')

# Now you can use it to classify images
print("Classifying images with loaded model...")

# Test multiple images
test_images = [
    'image1.jpg',
    'image2.jpg', 
    'image3.jpg'
]

for image_path in test_images:
    try:
        result = classifier.predict(image_path)
        print(f"{image_path} -> {result}")
    except FileNotFoundError:
        print(f"Could not find {image_path}")

print("All done!")
