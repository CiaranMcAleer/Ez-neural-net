"""
Robust Tests for Ez Vision
Tests the complete workflow with real data and models.
No mocks - actual training and prediction testing.
"""

import os
import sys
import shutil
import tempfile
import numpy as np
from PIL import Image, ImageDraw
import unittest
import json

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ez_vision import ImageClassifier, quick_classify, help
from nn_utils import train_neural_network, validate_input_data


class TestDataGenerator:
    """Generate synthetic test images for testing"""
    
    @staticmethod
    def create_test_image(size=(224, 224), color='red', shape='circle', filename=None):
        """Create a synthetic test image with specific characteristics"""
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        
        center = (size[0] // 2, size[1] // 2)
        radius = min(size) // 4
        
        # Color mapping
        colors = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128)
        }
        
        fill_color = colors.get(color, (128, 128, 128))
        
        if shape == 'circle':
            draw.ellipse([
                center[0] - radius, center[1] - radius,
                center[0] + radius, center[1] + radius
            ], fill=fill_color)
        elif shape == 'square':
            draw.rectangle([
                center[0] - radius, center[1] - radius,
                center[0] + radius, center[1] + radius
            ], fill=fill_color)
        elif shape == 'triangle':
            draw.polygon([
                (center[0], center[1] - radius),
                (center[0] - radius, center[1] + radius),
                (center[0] + radius, center[1] + radius)
            ], fill=fill_color)
        
        if filename:
            img.save(filename, 'JPEG')
        
        return img
    
    @staticmethod
    def create_test_dataset(base_path, categories, images_per_category=5):
        """Create a complete test dataset with multiple categories"""
        os.makedirs(base_path, exist_ok=True)
        
        # Define characteristics for each category
        category_configs = {
            'red_circles': {'color': 'red', 'shape': 'circle'},
            'blue_squares': {'color': 'blue', 'shape': 'square'},
            'green_triangles': {'color': 'green', 'shape': 'triangle'},
            'cats': {'color': 'yellow', 'shape': 'circle'},  # Mock cat images
            'dogs': {'color': 'purple', 'shape': 'square'},  # Mock dog images
        }
        
        created_files = []
        
        for category in categories:
            category_path = os.path.join(base_path, category)
            os.makedirs(category_path, exist_ok=True)
            
            config = category_configs.get(category, {'color': 'red', 'shape': 'circle'})
            
            for i in range(images_per_category):
                filename = os.path.join(category_path, f'{category}_{i+1}.jpg')
                TestDataGenerator.create_test_image(
                    size=(224, 224),
                    color=config['color'],
                    shape=config['shape'],
                    filename=filename
                )
                created_files.append(filename)
        
        return created_files


class TestEzVisionCore(unittest.TestCase):
    """Test core Ez Vision functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.training_dir = os.path.join(self.test_dir, 'training_data')
        self.model_dir = os.path.join(self.test_dir, 'test_model')
        
        # Create test dataset
        self.categories = ['red_circles', 'blue_squares']
        self.test_files = TestDataGenerator.create_test_dataset(
            self.training_dir, 
            self.categories, 
            images_per_category=8  # Increased for more reliable training
        )
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_image_classifier_creation(self):
        """Test ImageClassifier can be created with different parameters"""
        # Default parameters
        classifier1 = ImageClassifier()
        self.assertEqual(classifier1.image_size, (224, 224))
        self.assertFalse(classifier1.trained)
        
        # Custom image size
        classifier2 = ImageClassifier(image_size=(128, 128))
        self.assertEqual(classifier2.image_size, (128, 128))
    
    def test_image_classifier_training(self):
        """Test complete training workflow"""
        classifier = ImageClassifier()
        
        # Test training
        history = classifier.train(self.training_dir, epochs=1)  # Quick training
        
        # Verify training completed
        self.assertTrue(classifier.trained)
        self.assertEqual(len(classifier.class_names), 2)
        self.assertIn('red_circles', classifier.class_names)
        self.assertIn('blue_squares', classifier.class_names)
        
        # Verify history object
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
    
    def test_image_classifier_prediction(self):
        """Test prediction after training"""
        classifier = ImageClassifier()
        classifier.train(self.training_dir, epochs=1)
        
        # Create a test image
        test_image_path = os.path.join(self.test_dir, 'test_image.jpg')
        TestDataGenerator.create_test_image(
            color='red', shape='circle', filename=test_image_path
        )
        
        # Test prediction
        result = classifier.predict(test_image_path)
        self.assertIn(result, classifier.class_names)
        
        # Test prediction with confidence
        confidence_result = classifier.predict_with_confidence(test_image_path)
        self.assertIsInstance(confidence_result, dict)
        self.assertEqual(len(confidence_result), len(classifier.class_names))
        
        # Verify confidence scores sum to approximately 100%
        total_confidence = sum(confidence_result.values())
        self.assertAlmostEqual(total_confidence, 100.0, delta=5.0)
    
    def test_model_save_and_load(self):
        """Test saving and loading trained models"""
        # Train a model
        classifier1 = ImageClassifier()
        classifier1.train(self.training_dir, epochs=1)
        
        # Save the model
        classifier1.save(self.model_dir)
        
        # Verify files were created
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, 'model.keras')))
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, 'config.json')))
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, 'predict.py')))
        
        # Load the model in a new classifier
        classifier2 = ImageClassifier()
        classifier2.load(self.model_dir)
        
        # Verify loaded model has same properties
        self.assertTrue(classifier2.trained)
        self.assertEqual(classifier2.class_names, classifier1.class_names)
        self.assertEqual(classifier2.image_size, classifier1.image_size)
        
        # Test prediction with loaded model
        test_image_path = os.path.join(self.test_dir, 'test_image.jpg')
        TestDataGenerator.create_test_image(
            color='blue', shape='square', filename=test_image_path
        )
        
        result = classifier2.predict(test_image_path)
        self.assertIn(result, classifier2.class_names)
    
    def test_generated_prediction_script(self):
        """Test that the auto-generated prediction script works"""
        # Train and save a model
        classifier = ImageClassifier()
        classifier.train(self.training_dir, epochs=1)
        classifier.save(self.model_dir)
        
        # Test the generated prediction script
        script_path = os.path.join(self.model_dir, 'predict.py')
        self.assertTrue(os.path.exists(script_path))
        
        # Create a test image
        test_image_path = os.path.join(self.test_dir, 'script_test.jpg')
        TestDataGenerator.create_test_image(
            color='red', shape='circle', filename=test_image_path
        )
        
        # Import and use the generated script
        sys.path.insert(0, self.model_dir)
        try:
            import predict
            result = predict.predict_image(test_image_path)
            self.assertIn(result, classifier.class_names)
        finally:
            sys.path.remove(self.model_dir)
            if 'predict' in sys.modules:
                del sys.modules['predict']
    
    def test_quick_classify_function(self):
        """Test the quick_classify convenience function"""
        # Create test image
        test_image_path = os.path.join(self.test_dir, 'quick_test.jpg')
        TestDataGenerator.create_test_image(
            color='blue', shape='square', filename=test_image_path
        )
        
        # Test quick classify
        result = quick_classify(self.training_dir, test_image_path, epochs=1)
        self.assertIsInstance(result, str)
        self.assertIn(result, self.categories)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_training_with_nonexistent_folder(self):
        """Test error handling for nonexistent training folder"""
        classifier = ImageClassifier()
        
        with self.assertRaises(ValueError) as context:
            classifier.train('/nonexistent/folder')
        
        self.assertIn('not found', str(context.exception))
    
    def test_training_with_insufficient_categories(self):
        """Test error handling for insufficient categories"""
        # Create folder with only one category
        single_category_dir = os.path.join(self.test_dir, 'single_cat')
        TestDataGenerator.create_test_dataset(
            single_category_dir, ['red_circles'], images_per_category=5
        )
        
        classifier = ImageClassifier()
        
        with self.assertRaises(ValueError) as context:
            classifier.train(single_category_dir)
        
        self.assertIn('at least 2 categories', str(context.exception))
    
    def test_prediction_before_training(self):
        """Test error when trying to predict before training"""
        classifier = ImageClassifier()
        
        with self.assertRaises(ValueError) as context:
            classifier.predict('some_image.jpg')
        
        self.assertIn('train the model first', str(context.exception))
    
    def test_prediction_with_nonexistent_image(self):
        """Test error when predicting nonexistent image"""
        # Create and train a basic model
        training_dir = os.path.join(self.test_dir, 'training')
        TestDataGenerator.create_test_dataset(
            training_dir, ['red_circles', 'blue_squares'], images_per_category=5
        )
        
        classifier = ImageClassifier()
        classifier.train(training_dir, epochs=1)
        
        with self.assertRaises(ValueError) as context:
            classifier.predict('/nonexistent/image.jpg')
        
        self.assertIn('not found', str(context.exception))
    
    def test_loading_nonexistent_model(self):
        """Test error when loading nonexistent model"""
        classifier = ImageClassifier()
        
        with self.assertRaises(ValueError) as context:
            classifier.load('/nonexistent/model')
        
        self.assertIn('not found', str(context.exception))
    
    def test_saving_untrained_model(self):
        """Test error when saving untrained model"""
        classifier = ImageClassifier()
        
        with self.assertRaises(ValueError) as context:
            classifier.save(self.test_dir)
        
        self.assertIn('train the model first', str(context.exception))


class TestLegacyNeuralNetworkUtils(unittest.TestCase):
    """Test that original neural network utilities still work"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_validate_input_data(self):
        """Test input data validation"""
        # Test valid data
        input_values = [[1, 2], [3, 4], [5, 6]]
        target_values = [[10], [20], [30]]
        
        X, y = validate_input_data(input_values, target_values)
        
        self.assertEqual(X.shape, (3, 2))
        self.assertEqual(y.shape, (3, 1))
        
        # Test 1D input reshaping
        input_1d = [1, 2, 3]
        target_1d = [10, 20, 30]
        
        X, y = validate_input_data(input_1d, target_1d)
        
        self.assertEqual(X.shape, (3, 1))
        self.assertEqual(y.shape, (3, 1))
    
    def test_validate_input_data_errors(self):
        """Test error handling in input validation"""
        # Test empty data
        with self.assertRaises(ValueError):
            validate_input_data([], [1, 2, 3])
        
        with self.assertRaises(ValueError):
            validate_input_data([1, 2, 3], [])
    
    def test_neural_network_training(self):
        """Test complete neural network training workflow"""
        # Create simple training data (XOR-like problem)
        input_values = [
            [0, 0], [0, 1], [1, 0], [1, 1],
            [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]
        ]
        target_values = [
            [0], [1], [1], [0],
            [0], [1], [1], [0]
        ]
        
        output_path = os.path.join(self.test_dir, 'nn_model')
        
        # Train the network
        train_neural_network(
            input_values=input_values,
            target_values=target_values,
            output_path=output_path,
            max_training_time=30,  # Short training time for testing
            patience=3
        )
        
        # Verify output files were created
        self.assertTrue(os.path.exists(os.path.join(output_path, 'trained_model.keras')))
        self.assertTrue(os.path.exists(os.path.join(output_path, 'use_util.py')))
        
        # Test the generated utility
        sys.path.insert(0, output_path)
        try:
            from use_util import use_model
            
            # Test prediction
            test_input = [[0, 0], [1, 1]]
            predictions = use_model(test_input)
            
            self.assertIsInstance(predictions, list)
            self.assertEqual(len(predictions), 2)
        finally:
            sys.path.remove(output_path)
            if 'use_util' in sys.modules:
                del sys.modules['use_util']


class TestHelpAndUtilities(unittest.TestCase):
    """Test help functions and utility features"""
    
    def test_help_function(self):
        """Test that help function runs without errors"""
        # Capture output
        import io
        from contextlib import redirect_stdout
        
        output = io.StringIO()
        with redirect_stdout(output):
            help()
        
        help_text = output.getvalue()
        
        # Verify help content contains key information
        self.assertIn('IMAGE CLASSIFICATION', help_text)
        self.assertIn('QUICK MODE', help_text)
        self.assertIn('ImageClassifier', help_text)
        self.assertIn('quick_classify', help_text)


class TestIntegrationWorkflows(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_complete_computer_vision_workflow(self):
        """Test a complete computer vision workflow from start to finish"""
        # 1. Create training data
        training_dir = os.path.join(self.test_dir, 'cv_training')
        categories = ['cats', 'dogs']
        TestDataGenerator.create_test_dataset(
            training_dir, categories, images_per_category=8  # More images for stability
        )
        
        # 2. Train model
        classifier = ImageClassifier(image_size=(128, 128))  # Smaller for faster testing
        history = classifier.train(training_dir, epochs=2)
        
        # 3. Create test images
        cat_test = os.path.join(self.test_dir, 'cat_test.jpg')
        dog_test = os.path.join(self.test_dir, 'dog_test.jpg')
        
        TestDataGenerator.create_test_image(
            size=(128, 128), color='yellow', shape='circle', filename=cat_test
        )
        TestDataGenerator.create_test_image(
            size=(128, 128), color='purple', shape='square', filename=dog_test
        )
        
        # 4. Test predictions
        cat_prediction = classifier.predict(cat_test)
        dog_prediction = classifier.predict(dog_test)
        
        self.assertIn(cat_prediction, categories)
        self.assertIn(dog_prediction, categories)
        
        # 5. Test detailed predictions
        cat_confidence = classifier.predict_with_confidence(cat_test)
        self.assertEqual(len(cat_confidence), 2)
        
        # 6. Save model
        model_dir = os.path.join(self.test_dir, 'cv_model')
        classifier.save(model_dir)
        
        # 7. Load in new classifier and test
        new_classifier = ImageClassifier()
        new_classifier.load(model_dir)
        
        new_prediction = new_classifier.predict(cat_test)
        self.assertIn(new_prediction, categories)
        
        # 8. Test quick classify
        another_test = os.path.join(self.test_dir, 'quick_test.jpg')
        TestDataGenerator.create_test_image(
            size=(128, 128), color='yellow', shape='circle', filename=another_test
        )
        
        quick_result = quick_classify(training_dir, another_test, epochs=1)
        self.assertIn(quick_result, categories)


def run_tests():
    """Run all tests and provide detailed output"""
    print("🧪 Running Ez Vision Robust Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEzVisionCore,
        TestErrorHandling,
        TestLegacyNeuralNetworkUtils,
        TestHelpAndUtilities,
        TestIntegrationWorkflows
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed! Ez Vision is working perfectly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
