### MLP with MNIST
TensorFlow model to do digit classification using the MNIST data set. MNIST is a labeled set of images of handwritten digits.

##### Network Topology 

![][logo]

[logo]: https://github.com/Dinidu/MLP_EVALUATION/blob/master/docs/images/Topology.png?raw=true "Logo Title Text 2"

##### Design :
The above MLP topology is designed to require minimal preprocessing and it is called convolutional neural network (CNN) which is a class of deep, feed-forward artificial neural networks, most commonly applied to analyzing visual imagery. 

The above CNN consists of an input and an output layer, as well as multiple hidden layers. The hidden layers of the above CNN consist of convolutional layers, pooling layers, fully connected layers and normalization layers.

Hidden Layers :

* Two Convolutional Layers 
* Two Local Max Pooling Layers
* Two Fully Connected Layers

The raw input passes through several convolution and max pooling layers with rectified linear activations before several fully connected layers and a softmax loss for predicting the output class. During training, we use dropout.

For more details about the implementation goto: [ Implementaion ](https://www.google.com)


##### How to use above network :

###### Prerequisites:

This network has been developed and tested with the following configurations, Make sure that the following environment configurations are in place before running the network 

```
python version 3.5
pip version 18.1
virtualenv version 16.1.0
```
Once the above requirements are satisfied, run the following steps to get started.

Run
```
git clone git@github.com:Dinidu/MLP_EVALUATION.git
cd MLP_EVALUATION
pip install -r requirements.txt
```

As of now, the environment is ready to run the network, Before that make the desired parameters changes to change the network model. 

Following .py consist of all the parametes to evaluate the network.

```
util\constant.py
```
```python
DATA_PATH = 'dataset/mnist-data/'
IMAGE_SIZE = 28
PIXEL_DEPTH = 255

# Number of classes
NUM_CLASSES = 10
# Validation data size
VALIDATION_SIZE = 5000
# This defines the size of the batch.
BATCH_SIZE = 50
#one channel in grayscale images.
NUM_CHANNELS = 1
# The random seed that defines initialization.
SEED = 42

#Learning parameters
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DECAY_RATE = 0.95
DROP_OUT = True
L2_REGULARIZE = True
```
By making changes to the above properties, **Different versions of the network model** can be derived.








