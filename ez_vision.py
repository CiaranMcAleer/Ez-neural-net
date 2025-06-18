"""
Ez Vision - Simple Computer Vision for Everyone

Make computer vision models in just a few lines of code!
No ML experience required.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
import matplotlib.pyplot as plt

class ImageClassifier:
    """
    Easily train a model to recognize different types of images.
    
    Example:
        # Create a classifier
        classifier = ImageClassifier()
        
        # Train it on your images (organized in folders by category)
        classifier.train('my_photos/')
        
        # Use it to predict new images
        result = classifier.predict('new_photo.jpg')
        print(f"This looks like: {result}")
    """
    
    def __init__(self, image_size=(224, 224)):
        """
        Create a new image classifier.
        
        Args:
            image_size: How big to make images (width, height). Default is good for most cases.
        """
        self.image_size = image_size
        self.model = None
        self.class_names = []
        self.trained = False
        
    def train(self, folder_path, epochs=10, test_split=0.2):
        """
        Train the classifier on your images.
        
        Organize your images like this:
        my_photos/
            cats/
                cat1.jpg
                cat2.jpg
            dogs/
                dog1.jpg
                dog2.jpg
            birds/
                bird1.jpg
                bird2.jpg
        
        Args:
            folder_path: Path to folder containing your image categories
            epochs: How many times to train (more = better but slower)
            test_split: How much data to save for testing (0.2 = 20%)
        """
        print("🚀 Starting to train your image classifier...")
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder '{folder_path}' not found!")
            
        # Get class names from folder structure
        self.class_names = [d for d in os.listdir(folder_path) 
                           if os.path.isdir(os.path.join(folder_path, d))]
        
        if len(self.class_names) < 2:
            raise ValueError("You need at least 2 categories of images!")
            
        print(f"📂 Found {len(self.class_names)} categories: {', '.join(self.class_names)}")
        
        # Prepare data
        datagen = ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values
            rotation_range=20,  # Randomly rotate images
            width_shift_range=0.2,  # Randomly shift images
            height_shift_range=0.2,
            horizontal_flip=True,  # Randomly flip images
            validation_split=test_split
        )
        
        # Load training data
        train_generator = datagen.flow_from_directory(
            folder_path,
            target_size=self.image_size,
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        # Load validation data  
        validation_generator = datagen.flow_from_directory(
            folder_path,
            target_size=self.image_size,
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        # Build the model using a pre-trained network (transfer learning)
        base_model = EfficientNetB0(
            weights='imagenet',  # Pre-trained on millions of images
            include_top=False,   # Remove the final classification layer
            input_shape=(*self.image_size, 3)
        )
        
        # Freeze the base model (don't retrain it)
        base_model.trainable = False
        
        # Add our own classification layers
        self.model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Configure the model for training
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("🎯 Training the model...")
        
        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            verbose=1
        )
        
        # Get final accuracy
        final_accuracy = history.history['val_accuracy'][-1] * 100
        print(f"✅ Training complete! Final accuracy: {final_accuracy:.1f}%")
        
        self.trained = True
        
        return history
    
    def predict(self, image_path):
        """
        Predict what category an image belongs to.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            The predicted category name
        """
        if not self.trained:
            raise ValueError("You need to train the model first! Call .train() method.")
            
        if not os.path.exists(image_path):
            raise ValueError(f"Image file '{image_path}' not found!")
            
        # Load and prepare the image
        img = load_img(image_path, target_size=self.image_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index] * 100
        
        predicted_class = self.class_names[predicted_class_index]
        
        print(f"🔍 Prediction: {predicted_class} ({confidence:.1f}% confident)")
        
        return predicted_class
    
    def predict_with_confidence(self, image_path):
        """
        Predict what category an image belongs to, with confidence scores.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction and confidence for each category
        """
        if not self.trained:
            raise ValueError("You need to train the model first! Call .train() method.")
            
        if not os.path.exists(image_path):
            raise ValueError(f"Image file '{image_path}' not found!")
            
        # Load and prepare the image
        img = load_img(image_path, target_size=self.image_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        
        # Create results dictionary
        results = {}
        for i, class_name in enumerate(self.class_names):
            confidence = predictions[0][i] * 100
            results[class_name] = round(confidence, 1)
            
        # Sort by confidence
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return results
    
    def save(self, save_path):
        """
        Save your trained model so you can use it later.
        
        Args:
            save_path: Where to save the model
        """
        if not self.trained:
            raise ValueError("You need to train the model first!")
            
        os.makedirs(save_path, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(save_path, 'model.keras')
        self.model.save(model_path)
        
        # Save the class names and settings
        config = {
            'class_names': self.class_names,
            'image_size': self.image_size
        }
        
        config_path = os.path.join(save_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        # Create an easy-to-use prediction script
        script_content = f'''"""
Easy prediction script for your trained model
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_image(image_path):
    """
    Predict what category an image belongs to.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        The predicted category name
    """
    # Load model and config
    script_dir = os.path.dirname(__file__)
    model = tf.keras.models.load_model(os.path.join(script_dir, 'model.keras'))
    
    with open(os.path.join(script_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    class_names = config['class_names']
    image_size = tuple(config['image_size'])
    
    # Load and prepare the image
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index] * 100
    
    predicted_class = class_names[predicted_class_index]
    
    print(f"Prediction: {{predicted_class}} ({{confidence:.1f}}% confident)")
    
    return predicted_class

# Example usage:
# result = predict_image('my_photo.jpg')
# print(f"This looks like: {{result}}")
'''
        
        script_path = os.path.join(save_path, 'predict.py')
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        print(f"💾 Model saved to: {save_path}")
        print(f"📝 Use predict.py to make predictions with your saved model!")
    
    def load(self, save_path):
        """
        Load a previously trained model.
        
        Args:
            save_path: Path where the model was saved
        """
        if not os.path.exists(save_path):
            raise ValueError(f"Save path '{save_path}' not found!")
            
        # Load the model
        model_path = os.path.join(save_path, 'model.keras')
        self.model = keras.models.load_model(model_path)
        
        # Load the config
        config_path = os.path.join(save_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        self.class_names = config['class_names']
        self.image_size = tuple(config['image_size'])
        self.trained = True
        
        print(f"📁 Model loaded! Can recognize: {', '.join(self.class_names)}")


class ObjectFinder:
    """
    Find and locate objects in images.
    
    Example:
        # Create an object finder
        finder = ObjectFinder()
        
        # Train it to find specific objects
        finder.train('my_object_photos/')
        
        # Find objects in new images
        objects = finder.find('new_photo.jpg')
        print(f"Found {len(objects)} objects!")
    """
    
    def __init__(self):
        """Create a new object finder."""
        print("🔍 Object detection coming soon!")
        print("For now, use ImageClassifier to recognize different types of images.")
        
    def train(self, folder_path):
        """Train the object finder (coming soon!)"""
        raise NotImplementedError("Object detection is coming in the next update!")
        
    def find(self, image_path):
        """Find objects in an image (coming soon!)"""
        raise NotImplementedError("Object detection is coming in the next update!")


def quick_classify(folder_path, test_image, epochs=10):
    """
    Super quick way to train and test an image classifier in one line!
    
    Args:
        folder_path: Path to your training images (organized in category folders)
        test_image: Path to image you want to classify
        epochs: How many times to train
        
    Returns:
        The predicted category
    """
    print("🚀 Quick classify mode!")
    
    # Create and train classifier
    classifier = ImageClassifier()
    classifier.train(folder_path, epochs=epochs)
    
    # Make prediction
    result = classifier.predict(test_image)
    
    return result


def help():
    """Show help and examples for using Ez Vision"""
    help_text = """
🎨 Ez Vision - Computer Vision Made Simple!

==== IMAGE CLASSIFICATION ====
Teach your computer to recognize different types of images:

    from ez_vision import ImageClassifier
    
    # Create classifier
    classifier = ImageClassifier()
    
    # Train on your images (put images in category folders)
    classifier.train('my_photos/')
    
    # Predict new images
    result = classifier.predict('test_photo.jpg')
    print(f"This is a: {result}")
    
    # Save your trained model
    classifier.save('my_model/')

==== QUICK MODE ====
Do everything in one line:

    from ez_vision import quick_classify
    
    result = quick_classify('training_photos/', 'test_photo.jpg')

==== FOLDER STRUCTURE ====
Organize your training images like this:

    my_photos/
        cats/
            cat1.jpg
            cat2.jpg
        dogs/
            dog1.jpg
            dog2.jpg
        birds/
            bird1.jpg
            bird2.jpg

==== TIPS ====
• Use at least 20 images per category for better results
• More training epochs = better accuracy but takes longer
• JPG, PNG, and most image formats work
• The model will automatically resize your images

Happy coding! 🚀
"""
    print(help_text)
