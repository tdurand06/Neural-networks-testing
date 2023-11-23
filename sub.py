import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

#Data set 1: fashion_MNIST

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
    return (x.astype('float32')-np.mean(x))/np.std(x)
x_train_norm = normalize(x_train)
x_test_norm = normalize(x_test)

# plot the class distribution
image_labels = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt",
                7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
test_label_counts = {}  # Initialize an empty dictionary to store counts
for label in test_labels:
    if label in test_label_counts:
        test_label_counts[label] += 1
    else:
        test_label_counts[label] = 1
test_counts = list(test_label_counts.values())  # Extract the count values and convert to a list

train_label_counts = {}  # Initialize an empty dictionary to store counts
for label in train_labels:
    if label in train_label_counts:
        train_label_counts[label] += 1
    else:
        train_label_counts[label] = 1
train_counts = list(train_label_counts.values())  # Extract the count values and convert to a list

plt.figure(figsize=(10,6))
plt.bar(list(image_labels.values()), train_counts)
plt.title("Training Data Class Distribution")
plt.ylabel("Count")
plt.xlabel("Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.bar(list(image_labels.values()), test_counts)
plt.title("Test Data Class Distribution")
plt.ylabel("Count")
plt.xlabel("Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot some images
for i in range(9):
 plt.subplot(330 + 1 + i)
 plt.imshow(train_images[i], cmap=plt.get_cmap('gray'))
plt.title("Example of photos of Fashion_MNIST data")
plt.show()


#Dataset 2: CIFAR 10
(train_images_2, train_labels_2), (test_images_2, test_labels_2) = tf.keras.datasets.cifar10.load_data()
# 32x32 pixel images with 3 color channels (RGB)
print("train_images:", train_images_2.shape)
print("test_images:", test_images_2.shape)

# Flatten the images
train_images_2_flattened = train_images_2.reshape(train_images_2.shape[0], -1)
test_images_2_flattened = test_images_2.reshape(test_images_2.shape[0], -1)
print("train_images after vectorizing:", train_images_2_flattened.shape)
print("test_images after vectorizing:", test_images_2_flattened.shape)

# Normalize
train_images_norm_2 = (train_images_2_flattened.astype('float32') - np.mean(train_images_2_flattened)) / np.std(
    train_images_2_flattened)
test_images_norm_2 = (test_images_2_flattened.astype('float32') - np.mean(test_images_2_flattened)) / np.std(
    test_images_2_flattened)

def one_hot_encode(y):
  class_labels = np.unique(y)
  one_hot_y = np.zeros((len(y), len(class_labels)))
  for i in range(len(y)):
    one_hot_y[i, y[i]] = 1
  return one_hot_y

# Encode the labels
y_train_2 = one_hot_encode(train_labels_2)
y_test_2 = one_hot_encode(test_labels_2)
# print(y_test)

label_names = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

print("Test Data Class Distribution")
labels, counts = np.unique(test_labels_2, return_counts=True)
# Create a dictionary for label counts
test_label_counts = dict(zip(label_names.values(), counts))
print(test_label_counts)

print("Training Data Class Distribution")
labels_TRAIN, counts_TRAIN = np.unique(train_labels_2, return_counts=True)
# Create a dictionary for label counts
train_label_counts = dict(zip(label_names.values(), counts_TRAIN))
print(train_label_counts)

plt.figure(figsize=(10, 6))
plt.bar(test_label_counts.keys(), test_label_counts.values())
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Test Data Class Distribution')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(train_label_counts.keys(), train_label_counts.values())
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Training Data Class Distribution')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Visualize
plt.figure()
plt.title("Example of an image labeled bird")
plt.imshow(train_images_2[123])
plt.colorbar()
plt.grid(False)
plt.show()

