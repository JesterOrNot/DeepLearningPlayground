# Deep Learning

My deep learning  notes

## Terminology

| Term | Definition |
|:----:|:----------:|
| ANN  |  Artificial Neural Network  |
| Layer | Organized Neurons |
| Deep Network | A Network with more than one hidden layer |
| Activation Function | Non linear function that follows a layer |
| Loss | Error between model's prediction and actual output |
| Epoch | A pass of data through the model during training |
| Gradient | Derivitive of a function with several variables. |

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

So how would we implement such a network in Keras? The following is a Keras implementation of the Neural network above

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(4, input_shape=(3,)),
    Dense(2)
])
```

The output of one layer is then passed to another and another, and soo on the output is computed with the following equation.


```python
output = activation(sum(weights))
```

this process is repeated over and over untill we reach the output layer. During this process weights will mutate in order to acheive optimized weighting for each connection. This is known as a foward pass.

### Activation Function

A activation function of a neuron defines  the output of the specified neuron given a set of inputs. Follows a layer. They are inspired by activity in the human brain where neurons fire or are activated by stimuli (i.e. When you smell something good specific neurons will fire) A neuron can either be firing represented by a 1 or not firing represented by a 0.

Here are some exmaples of activation functions

#### Sigmoid

Takes a value and if the number is very negative then sigmoid transforms the input to something close to 0, if it is very positive it will transform the input to something close to 1. And if it is 0 then it will transform the input into a value between 0 and 1.

Formula

<img src="https://latex.codecogs.com/svg.latex?\Large&space;sigmoid(x) = \dfrac{e^x}{e^x+1}" />


Python code

```python
from math import e

def sigmoid(x):
    return e**x / e**(x+1) 
```

#### Rectified Linear Unit (ReLU)

Transforms the input to the maxiumum of 0 and the input. It is one of the most widley used activation functions today.

Formula

<img src="https://latex.codecogs.com/svg.latex?\Large&space;relu(x) = \max(0,x)" />

Python code

```python
def relu(x):
    return max(0, x)
```

## Training A Model.

This is the part where our model  will learn. During training we supply our model with data (i.e. if we are making a model that classifies images of cats and dogs the data would be images of cats and dogs). That will be passed through the model and the output layer (in a classification problem) will give you probabilities that the input is something. And after multiple epochs the model would learn. 

### Optimization

When training a model basically were solving  a optimization problem. The things we're optimizing are the weights within the model. The way its optimized depends on the optimization algorithms. The job of the optimization problem being to reduce *loss*. Now let's look at some optimization algorithms.

### Loss

Loss is basically the error between what the model thinks the awnser is vs the actual awnser

### Learning

Here is the process off learning during training.

Learning rate is usually a value from 0.1 to 0.0001.

1. Pass Data through the model.
2. Compute the loss of the output.
3. Calculate the gradient of the loss function with respect to the *weights* within the network.
4. Multiply by the learning rate.

In Keras learning happens with the fit method and is applied with the various paramaters: x which is the data our model is trained on, y which are the corresponding labels to the data, batch_size which is the ammount of data passed to the model at one time, epochs the number of passes of the data to the model, shuffle (boolean) whether to shuffle the data on each pass, verbosity level of verbosity of output. An example of a call to fit would be:

```python
model.fit(
    x=scaled_train_samples, 
    y=train_labels, 
    batch_size=10, 
    epochs=20, 
    shuffle=True, 
    verbose=2
)
```

### Example

Say we have a program that we want to identify if the image is of a cat or a dog. We would have 2 output nodes One representing a cat. And the other a dog.

# Keras

Keras is a simple API for describing neural networks

## Layers

### Dense

This is the most basic layer in a neural network. It connects it's inputs to it's outputs. This layer merely connects inputs to outputs within it's layer.

### Convolutional

Used for work with images

### Recurrent

Used for work with time series data.

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
