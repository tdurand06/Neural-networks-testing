import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.utils
import tensorflow as tf
from scipy.special import softmax as sfm
plt.rcParams['image.cmap'] = 'Greys'
from tensorflow.keras.utils import to_categorical

# load the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# flatten the images
train_images_flattened = train_images.reshape(60000, 784)
test_images_flattened = test_images.reshape(10000, 784)

# normalize the images
x_train = (train_images_flattened.astype('float32') - np.mean(train_images_flattened)) / np.std(
    train_images_flattened)
x_test = (test_images_flattened.astype('float32') - np.mean(test_images_flattened)) / np.std(
    test_images_flattened)

# encode the labels
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

def normalize(x):
    return (x.astype('float32') - np.mean(x)) / np.std(x)

x_train_norm = normalize(x_train)
x_test_norm = normalize(x_test)

# plot the class distribution
image_labels = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt",
                7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}


class Layer:
    def __init__(self, input_size, output_size, activation):
        self.bias = np.zeros((1, output_size))  # might not need this
        self.weights = np.zeros((input_size, output_size))
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.input = None
        self.output = None

        self.dy = None
        self.dw = None
        self.b = None

    def init_weights(self, input_size, output_size, weight_pattern="Uniform"):
        tmp_weights = []
        # create weights (+1 is for bias) don't think i need this
        match (weight_pattern):
            case "Zeros":
                tmp_weights = (np.zeros((output_size, input_size)))
            case "Uniform":
                tmp_weights = (np.random.uniform(-1, 1, (output_size, input_size)))
            case "Gaussian":
                tmp_weights = (np.random.normal(0, 1, (output_size, input_size)))
            case "Xavier":
                tmp_weights = (
                    np.random.uniform(-1 / np.sqrt(input_size), 1 / np.sqrt(input_size), (output_size, input_size)))
            case "Kaiming":
                tmp_weights = (np.random.normal(0, 2 / input_size, (output_size, input_size)))
        inp_weights = np.array(tmp_weights)

        return np.array(inp_weights)

    def act_func(self, x):
        func_deriv = 0
        if self.activation == "relu":
            x[x < 0] = 0
            func_deriv = (x > 0) * 1
        elif self.activation == "leaky_relu":
            gamma = 0.01
            np.where(x > 0, x, x * gamma)
            func_deriv = np.where(x > 0, 1, gamma)
        elif self.activation == "tan_h":
            x = np.tanh(x)
            func_deriv = 1 - x ** 2
        return x, func_deriv

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(np.exp(x[i]) / np.sum(np.exp(x[i])))
        return np.array(tmp)


class Mlp:
    def __init__(self, input_size, hidden_layer_sizes, output_size, activation, learning_rate, batch_size,
                 weight_pattern, regulizer="", L1=0.0, L2=0.0):
        self.validate_acc_his = []
        self.cross_entropies = []
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iters = 1e2
        self.epsilon = 10e-7
        self.weight_pattern = weight_pattern
        self.layers = []
        self.datasize = 0
        self.regulizer = regulizer
        self.L1 = L1
        self.L2 = L2
        # layer size is: number of pixels, number of units in hidden layer, number of units in hidden layer, and 10 classes (or outputs)
        self.layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        for i in range(1, len(self.layer_sizes)):
            self.layers.append(Layer(self.layer_sizes[i - 1], self.layer_sizes[i], activation))

        # initialization:
        for i in range(0, len(self.layer_sizes) - 1):
            layer = self.layers[i]  # current layer
            layer.weights = layer.init_weights(layer.input_size, layer.output_size, self.weight_pattern).T
            # print(layer.weights)

    def forwardprop(self, X):

        for i in range(len(self.layers)):
            layer = self.layers[i]  # current layer
            if i == 0:  # input data
                layer.input = X
            else:
                layer.input = self.layers[i - 1].output
            if layer.output_size == self.output_size:
                layer.output = sfm(np.dot(layer.input, layer.weights) + layer.bias, axis=1)
            else:
                layer.output = layer.act_func((np.dot(layer.input, layer.weights) + layer.bias))[
                    0]  # the zero is to get the function not the derivative of the act func

    def backwardprop(self, Y):
        H = len(self.layer_sizes) - 2
        for L in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[L]
            if L == H:
                output_predicted = layer.output
                layer.dy = output_predicted - Y
                layer.dw = np.dot(layer.input.T, layer.dy) / Y.shape[
                    1]  # .shape[1] gives you columns, i.e. num of examples
            else:
                layer.dy = np.dot(self.layers[L + 1].dy, self.layers[L + 1].weights.T)
                layer.dw = np.dot(layer.input.T, layer.dy * layer.act_func(layer.output)[1]) / Y.shape[1]
            # print(layer.dw)
            if self.regulizer == "L1":
                layer.dw += np.sign(layer.weights) * self.L1
            elif self.regulizer == "L2":
                layer.dw += layer.weights * self.L2
            # print(self.datasize)
            layer.dw = layer.dw / self.input_size

    def cross_entropy(self, predicted, actual):
        ce = np.sum(actual * np.log(predicted + 1e-9))
        return -ce

    def accuracy(self, y, y_pred):
        correct = 0
        for i in range(len(y)):
            if np.argmax(y_pred[i]) == np.argmax(y[i]):
                correct += 1
        return correct / len(y)

    def gradientdescent(self, X, Y, NORMS):

        norms = NORMS
        self.forwardprop(X)
        self.backwardprop(Y)

        error = self.cross_entropy(self.layers[-1].output, Y)
        for i, layer in enumerate(self.layers):
            layer.weights -= self.learning_rate * layer.dw
            norms[i] = np.linalg.norm(layer.dw)
        return norms

    def fit(self, X, Y, num_iterations):
        validate_acc_his = []
        cross_entropies = []
        t = 1
        self.max_iters = num_iterations
        self.datasize = X.shape[0]
        norms = np.array([np.inf] * len(self.layers))
        while np.any(norms > self.epsilon) and t < self.max_iters:  # Check if all norms are smaller than epsilon
            mini_batches = []
            X_s, Y_s = sklearn.utils.shuffle(X, Y, random_state=t)
            for k in range(0, X_s.shape[0], self.batch_size):
                mini_batch = (X_s[k:k + self.batch_size], Y_s[k:k + self.batch_size])
                mini_batches.append(mini_batch)
            for mini_batch in mini_batches:
                norms = self.gradientdescent(mini_batch[0], mini_batch[1], norms)
            y_pred = self.predict(X)
            accuracy = self.accuracy(Y, y_pred)
            print("accuracy ", t, "is :", accuracy, "\n and norm is: ", norms)
            cross_entropy = self.cross_entropy(y_pred, Y)
            cross_entropies.append(cross_entropy)
            validate_acc_his.append(accuracy)
            t += 1
        print("t is: ", t, " and max iteration is: ", self.max_iters, "\n and norms is: ", norms)
        self.validate_acc_his = validate_acc_his
        self.cross_entropies = cross_entropies

    def predict(self, X):
        self.forwardprop(X)
        for i in range(len(self.layers)):
            layer = self.layers[i]

        return self.layers[-1].output


start = time.time()
mlp = Mlp(input_size=784, hidden_layer_sizes=[128, 128], output_size=10, activation="relu", learning_rate=0.3,
          batch_size=128, weight_pattern="Uniform", L2=0.1, regulizer="L2")
mlp.fit(x_train, y_train, num_iterations=100)
y_pred = mlp.predict(x_test)
accuracy = mlp.accuracy(y_test, y_pred)
end = time.time()
print("run time: ", end - start)
print(accuracy)


