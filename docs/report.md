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
pip version 1.8
```




