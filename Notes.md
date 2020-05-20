# Deep Learning

My deep learning  notes

## Terminology

| Term | Definition |
|:----:|:----------:|
| ANN  |  Artificial Neural Network  |
| Layer | Organized Neurons |
| Deep Network | A Network with more than one hidden layer |

### Other notes

model = net = neural network

neuron = node

## Layers

Layers are organized neurons.

Layers are sperated into 3 catagories

  - An input layer
  - one or more hidden layers
  - An output layer

Here is an image visualizing a network with s input layer made of 3 neurons, 1 hidden layer made up of 4 neurons, and a output layer made out of 2 neurons.

![An image of a neural network](images/neural-net.jpg)

Differnt layers will perform differnt operations on data. Data flows from the input layer through the hidden layers until the output layer is reached.

How many neurons should I assign to each layer?

| Type | How to allocate |
|:----:|:----:|
| Input | One neuron for each component of input data |
| Hidden | Arbitrarily Chosen |
| Output | One node for each of the possible desired outcomes |

So how would we implement such a network in Kera
s?

```python
from keras.models import Sequential

model = Sequential([
    Dense(units=3, input_shape)
])
```

# Keras

Keras is a simple API for describing neural networks

## Layers

### Dense

This is the most basic layer in a neural network.

## The Sequential Model

The sequential Model is a linear stack of layers i.e.

```python
# Import the Sequential Model
from keras.models import Sequential

model = Sequential([
    Dense(32, input_shape=(10,), activation='relu'),
    Dense(2, activation='softmax')
])
```