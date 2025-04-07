from nn_utils import train_image_classifier, preprocess_images
from nn_utils import predict

image_paths = [...]  # list of paths to your images
labels = [...]  # list of corresponding labels (e.g. integers or strings)

X, y = preprocess_images(image_paths, labels)
train_image_classifier(
    image_paths,
    labels,
    output_path="image_classifier_model",
    img_size=(64, 64),
    max_training_time=300,
    patience=5
)

#Use the model to make predictions
new_image_path = [...]  # path to a new image to classify
new_image = tf.keras.preprocessing.image.load_img(new_image_path, target_size=(64, 64))
new_image = tf.keras.preprocessing.image.img_to_array(new_image)
new_image = np.array([new_image])  # add batch dimension

prediction = predict(new_image)
print(prediction)