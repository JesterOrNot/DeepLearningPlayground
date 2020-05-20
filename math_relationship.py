import numpy as np
from tensorflow import keras

# This line declares the model.
# A model is a trained neural network.
model = keras.Sequential(
    # This declares a single layer for our network
    # This layer has a single neuron in it because units=1
    # Input shape means that we will be inserting one value
    # in this case we plug in a single x value
    keras.layers.Dense(units=1, input_shape=[1])
)
model.compile(
    # Generates guesses
    optimizer='sgd',
    # Use mean squared error to calculate how good the model is
    loss='mean_squared_error'
)


##### 2x-1 #####

# X inputs all of the float type
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

# Y inputs all of the float type
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model.
model.fit(
    xs,
    ys,
    # loop 500 times each time optimizing the model
    epochs=500
)

# Model 2

model2 = keras.Sequential(keras.layers.Dense(units=1, input_shape=[1]))
model2.compile(optimizer='sgd', loss="mean_squared_error")

###### 3x + 5 ##########

x1s = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y1s = np.array([2, 5, 8, 11, 14, 17], dtype=float)

model2.fit(
    x1s,
    y1s,
    epochs=500
)

print(model.predict([10, 11]))
print(model2.predict([16]))
