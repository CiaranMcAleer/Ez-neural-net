"""
Neural Network Utilities

This module provides functions for training and using a neural network model.
"""

import os
import time
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping

class TimeStopping(tf.keras.callbacks.Callback):
    """
    Custom callback to stop training once a certain amount of time (in seconds) has passed.
    """
    def __init__(self, model, max_seconds=None):
        """
        Initialize the callback.
        """
        super().__init__()
        self.model = model
        self.max_seconds = max_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        """
        Start the timer when training begins.
        """
        self.start_time = time.time()
        self.model = self.get_model() 

    def on_epoch_end(self, epoch, logs=None):
        """
        Check if the time limit has been reached and stop training if so.
        """
        if self.max_seconds is None:
            return  # no time limit
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            print(f"Max training time of {self.max_seconds} seconds reached. Stopping training.")
            self.model.stop_training = True

def validate_input_data(input_values, target_values):
    """Validate and prepare input data"""
    if not input_values or not target_values:
        raise ValueError("Input or target values cannot be empty")
    
    X = np.array(input_values)
    y = np.array(target_values)
    
    # If 1D array, reshape to 2D
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
        
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    return X, y

def preprocess_images(image_paths, labels, img_size=(64, 64)):
    """
    Preprocesses the images by resizing and normalizing them.

    Parameters:
    -----------
    image_paths : list of str
        Paths to the images.
    labels : list of int
        Corresponding labels for the images.
    img_size : tuple
        The size to which the images will be resized.

    Returns:
    --------
    X : np.array
        Preprocessed images.
    y : np.array
        Corresponding labels.
    """
    X = []
    for path in image_paths:
        img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        X.append(img)

    X = np.array(X) / 255.0  # Normalize pixel values to [0, 1]
    y = tf.keras.utils.to_categorical(labels, num_classes=len(set(labels)))

    return X, y

def train_image_classifier(
    image_paths,
    labels,
    output_path,
    img_size=(64, 64),
    max_training_time=300,
    patience=5
):
    """
    Train an image classifier model.

    Parameters:
    -----------
    image_paths : list of str
        Paths to the images.
    labels : list of int
        Corresponding labels for the images.
    output_path : str
        Path to the folder where the trained model will be saved.
    img_size : tuple
        The size to which the images will be resized.
    max_training_time : int, optional
        The maximum amount of time (in seconds) to train for. If None, no time limit is applied.
        Defaults to 300 seconds.
    patience : int, optional
        The number of epochs to wait before stopping training when no improvement is observed.
        Defaults to 5 epochs.

    Returns:
    -------
    None
    """
    try:
        # Preprocess data
        X, y = preprocess_images(image_paths, labels, img_size)

        # Ensure model output path exists
        os.makedirs(output_path, exist_ok=True)

        # Build a CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(img_size[0], img_size[1], 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(set(labels)), activation='softmax')  # Output layer for classification
        ])

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',  # Use categorical cross-entropy for classification
            metrics=['accuracy']
        )

        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',  # Monitor validation loss for early stopping
                patience=patience,
                restore_best_weights=True
            ),
            TimeStopping(max_seconds=max_training_time)
        ]

        print("Starting training...\n")
        # Fit the model
        history = model.fit(
            X,
            y,
            epochs=1000,            # Arbitrary high epoch count—will stop early if needed
            verbose=1,              # Print training progress each epoch
            callbacks=callbacks,
            validation_split=0.2   # Use a validation split for monitoring
        )
        print("\nTraining complete.")

        # Save the trained model into the specified folder.
        saved_model_path = os.path.join(output_path, "trained_model")
        model.save(saved_model_path)

        print(f"Model saved to {saved_model_path}")

        # Create the "use_util.py" script for inference
        use_util_path = os.path.join(output_path, "use_util.py")
        with open(use_util_path, "w", encoding="utf-8") as f:
            f.write(f"""import os
import numpy as np
import tensorflow as tf

def use_model(image_paths, img_size=(64, 64)):
    \"""
    Loads the trained model from the local 'trained_model' folder
    and runs inference on the provided image paths.

    Parameters:
    -----------
    image_paths : list of str
        Paths to the images.
    img_size : tuple
        The size to which the images will be resized.

    Returns:
    --------
    A Python list containing the model's predictions.
    \"""
    # Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), "trained_model")
    model = tf.keras.models.load_model(model_path)

    # Preprocess images
    X = []
    for path in image_paths:
        img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        X.append(img)

    X = np.array(X) / 255.0  # Normalize pixel values to [0, 1]

    # Run predictions
    preds = model.predict(X)

    # Convert predictions to a plain Python list before returning
    return preds.tolist()
""")

        print(f"'use_util.py' created at {use_util_path}")

        # Print final training stats
        final_loss = history.history["loss"][-1]
        final_accuracy = history.history["accuracy"][-1]
        print(f"Final training loss: {final_loss:.6f}")
        print(f"Final training accuracy: {final_accuracy:.6f}")
        if max_training_time:
            print(f"Training was time-limited to {max_training_time} seconds (or until early stopping).")
        else:
            print(f"Training ended via EarlyStopping after {len(history.history['loss'])} epochs.")

        print("\nAll done! You can now import `use_util.py` and call `use_model(...)` for predictions.")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def predict(input_list):
    """Make predictions with validation"""
    try:
        X = np.array(input_list)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), "trained_model"))
        return model.predict(X).tolist()
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise
