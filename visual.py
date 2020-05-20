from keras.models import Sequential
from keras.layers import Dense
from ann_visualizer.visualize import ann_viz;

model = Sequential([
    Dense(4, input_shape=(3,)),
    Dense(2)
])
ann_viz(model, title="My first neural network", view=True)
