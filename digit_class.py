import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# setup network architecture
network = keras.models.Sequential()
network.add(keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(keras.layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# data preprocessing, change data into float32 array with values between 0 and 1
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# encode labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# train network
network.fit(train_images, train_labels, epochs=10, batch_size=64)

# evaluate network
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test accuracy: ', test_acc)
