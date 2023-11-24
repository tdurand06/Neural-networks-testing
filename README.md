# Neural-networks-testing
This repository concerns itself with the classification of images using two techniques: a multi-linear perceptron (MLP) coded from scratch (found in main.py), and a tuned CNN that leverages torch.nn (found in CNN.py)

# Data
The data used for the MLP is fasion_MNIST, a collection of 70,000 images split into a 60,000 training and 10,000 test images. The data used for the CNN is CIFAR 10, a collection of 60,000 images split into 50,000 training and 10,000 testing images. The Fasion_MNIST dataset, its downloadable link and information about it can be found [here](https://www.kaggle.com/datasets/zalando-research/fashionmnist). More conveniently, it can be downloaded using the tensorflow library:
`import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
`
Information about the CIFAR 10 data can be found [here](https://www.cs.toronto.edu/~kriz/cifar.htm)

Please view `sub.py` for more info about both these datasets. 

# Files

### main.py
Contains code that imports and preprocesses the Fashion_MNIST data.
It then runs a multi-linear perceptron algorithm implemented with mini batch stochastic gradient descent. 
The accuracy is around 85%. 

### sub.py
Contains code that imports, preprocesses and vizualizes both the images and the distribution of the Fashion_MNIST and CIFAR_10 Data. 

### CNN.py
Contains code to implement a CNN from torch.nn, with fine-tuned hyperparameters.

# Conclusion:

Therefore, the CNN model was found to be optimal for image classification tasks. This was attributed to the ability of the CNN model to take advantage of the underlying correlations between image pixels. 


