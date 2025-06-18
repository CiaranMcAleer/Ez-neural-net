"""
Example 1: Basic Image Classification

This example shows how to train a model to recognize different types of images.
Perfect for beginners!
"""

from ez_vision import ImageClassifier

# Step 1: Create an image classifier
print("Creating image classifier...")
classifier = ImageClassifier()

# Step 2: Train it on your images
# Make sure your images are organized like:
# my_photos/
#     cats/
#         cat1.jpg, cat2.jpg, etc.
#     dogs/  
#         dog1.jpg, dog2.jpg, etc.
#     birds/
#         bird1.jpg, bird2.jpg, etc.

print("Training the model...")
classifier.train('my_photos/', epochs=15)  # Train for 15 rounds

# Step 3: Test it on a new image
print("Testing on a new image...")
result = classifier.predict('test_image.jpg')
print(f"The model thinks this image shows: {result}")

# Step 4: Get detailed predictions
print("Getting detailed predictions...")
detailed = classifier.predict_with_confidence('test_image.jpg')
print("Confidence for each category:")
for category, confidence in detailed.items():
    print(f"  {category}: {confidence}%")

# Step 5: Save the model for later use
print("Saving the model...")
classifier.save('my_trained_model/')
print("Done! You can use this model later.")
