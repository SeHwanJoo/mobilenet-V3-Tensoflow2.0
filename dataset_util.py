import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100


def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images


def load_images():
    (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    (train_images, test_images) = normalization(train_images, test_images)

    train_labels = to_categorical(train_labels, 100)
    test_labels = to_categorical(test_labels, 100)

    return train_images, train_labels, test_images, test_labels


def build_optimizer(learning_rate=0.1, momentum=0.9):
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [58500, 97500],
        [learning_rate, learning_rate / 10., learning_rate / 100.])

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

    return optimizer
