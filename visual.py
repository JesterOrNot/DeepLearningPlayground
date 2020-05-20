from ann_visualizer.visualize import ann_viz
from keras.layers import Dense
from keras.models import Sequential

# Input Layer: 3 Nodes
# Hidden Layer: 4 Nodes
# Output Layer: 2 Nodes
model = Sequential([
    Dense(4, input_shape=(3,)),
    Dense(2)
])
ann_viz(model, title="My first neural network", view=True)
