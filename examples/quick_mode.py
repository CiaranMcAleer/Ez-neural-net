"""
Example 2: Super Quick Classification

This shows how to do everything in just a few lines!
Perfect when you want to quickly test something.
"""

from ez_vision import quick_classify

# Do everything in one line!
# This will:
# 1. Create a classifier
# 2. Train it on your photos
# 3. Test it on your image
# 4. Return the result

result = quick_classify(
    folder_path='my_photos/',     # Your training images
    test_image='test_image.jpg',  # Image to classify
    epochs=10                     # How many training rounds
)

print(f"Quick result: This image shows a {result}!")

# That's it! Super simple.
