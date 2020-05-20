import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.mnist

# Load dataset
# Images are the image of the item
# Labels are numbers which indicate what the image is of
(
(training_images, training_labels),
(test_images, test_labels)
) = fashion_mnist.load_data()


# Start with a 28x28 image and return a number
model = keras.Sequential(
[
    # Take a 28x28 image represented by a multi dimensional array
    # And make it into a 1D array
    keras.layers.Flatten(input_shape=(28, 28)),
    # Only pass values 0 or greater
    keras.layers.Dense(128, activation=tf.nn.relu),
    # Set the largest value to 1 and the rest to 0
    # Saves us from fishing out the largest value#
    keras.layers.Dense(10, activation=tf.nn.softmax),
]
)

model.compile(
    optimizer='adam', loss="sparse_categorical_crossentropy"
)

model.fit(training_images, training_labels, epochs=5)
classifications = model.predict([test_images[0]])
print(classifications[0])
