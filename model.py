from keras.models import Sequential
from keras.layers import Dense, Activation
from ann_visualizer.visualize import ann_viz;

model = Sequential([
    Dense(units=3, input_shape=(2,), activation='relu'),
    Dense(units=2, activation='softmax')
])
model.compile(
    # Generates guesses
    optimizer='sgd',
    # Use mean squared error to calculate how good the model is
    loss='mean_squared_error'
)
ann_viz(model, title="My first neural network")