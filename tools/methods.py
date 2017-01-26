import os
import importlib
import random
import time

import numpy as np
import sklearn

import keras
from keras.utils import np_utils
from keras.optimizers import Adagrad

import dataprocessing

def center_crop(data, new_dim):
    """
    Center crop a set of images.
    The data must have shape (n_samples, n_channels, width, height).
    """

    dim = data.shape[2]
    offset = (dim - new_dim) / 2
    return data[:, :, offset:offset+new_dim, offset:offset+new_dim]


def random_crop(img, new_dim):
    """
    Randomly crop an image.
    """

    dim = img.shape[1]
    offset = dim - new_dim
    
    idx = random.randint(0, offset)
    idy = random.randint(0, offset)

    new_img = img[:, idx:idx+new_dim, idy:idy+new_dim]

    if random.randint(0, 1) == 0:
        new_img = new_img[:, :, ::-1]

    return new_img


def augment(data, crop_dim):
    """
    Augment a dataset by random cropping.
    The data must have shape (n_samples, n_channels, width, height).
    """

    new_data = []
    for img in data:
        new_img = np.array(img)
        new_img = random_crop(new_img, crop_dim)
    
        new_data.append(new_img)
    return np.array(new_data)


class DeepLearning():
    """
    Represents a deep learning network using a given model.
    """

    def __init__(self, model):
        """Loads model"""

        self._model_module = importlib.import_module('models' + '.' + model)
        self._network = None
    
    @staticmethod
    def _iterate_minibatches(X, y, batch_size, shuffle=False, seed=42):
        """
        Generates data batches to feed into the network.
        """

        if shuffle:
            X, y = sklearn.utils.shuffle(X, y, random_state=seed)

        n_batches = X.shape[0] / batch_size
        for i in range(n_batches):
            idx = i * n_batches
            yield X[idx:idx+batch_size], y[idx:idx+batch_size]


    def train(self, dirname, n_epochs, batch_size=32, 
            target_size=(140, 140), crop_dim=128):
        """Train the classfier with the given model."""

        # split the data and determine components
        print "Splitting data ..."
        
        (X_train, y_train), (X_val, y_val), n_classes = dataprocessing.split_data(
                                            dirname, target_size, transpose=True)

        X_train = keras.backend.cast_to_floatx(X_train) / 255.0
        X_val = keras.backend.cast_to_floatx(X_val) / 255.0

        y_train = np_utils.to_categorical(y_train, n_classes)
        y_val = np_utils.to_categorical(y_val, n_classes)

        print ""
        print "Number of training samples:", X_train.shape, X_train.dtype
        print "Number of validation samples:", X_val.shape, X_val.dtype
        print y_train.shape

        # load and compile network
        input_shape = (3, crop_dim, crop_dim)
        self._network = network = self._model_module.build_model(
                                    input_shape, n_classes)

        optimizer = Adagrad(lr=0.009, epsilon=1e-03)
        network.compile(optimizer=optimizer, loss='categorical_crossentropy',
                        metrics=['accuracy'])

        print ""
        print "Begin training ..."
        best_val_acc = 0.0
        for epoch in range(n_epochs):
            # train on batches
            train_err = 0.0
            train_batches = 0
            start_time = time.time()

            for X_batch, y_batch in self._iterate_minibatches(X_train, y_train, 
                                                    batch_size, shuffle=True):
                X_augmented = augment(X_batch, crop_dim)
                batch_err, batch_acc = network.train_on_batch(X_augmented, 
                                                            y_batch)
                train_err += batch_err
                train_batches += 1

            X_val_cropped = center_crop(X_val, crop_dim)
            val_err, val_acc = network.evaluate(X_val_cropped, y_val, 
                                                batch_size=batch_size)
            best_val_acc = max(val_acc, best_val_acc)
            # print out information at the end of each epoch
            print "Epoch {}/{} took {:.3f}s: ".format(epoch, n_epochs,
                                                time.time()-start_time),
            print "Training loss = {},".format(train_err / train_batches),
            print "Validation loss = {},".format(val_err),
            print "Validation acc = {:.2f}%.".format(val_acc * 100)

        print "Best validation accuracy = {:.2f}".format(best_val_acc * 100)