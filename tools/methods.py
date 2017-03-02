import os
import sys
import importlib
import random
import time
import cPickle

import numpy as np
import sklearn

import keras
from keras.utils import np_utils
from keras.optimizers import Adagrad, Adadelta, Adam
from keras.wrappers.scikit_learn import KerasClassifier

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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


class DeepLearning():
    """
    Represents a deep learning network using a given model.
    """
    models_list = set(['conv3', 'resnet'])

    optimizers_list = {
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam
    }

    def __init__(self, model):
        """Loads model"""

        if model not in self.models_list:
            raise Exception("Model does not exist.")

        self._model_module = importlib.import_module('models' + '.' + model)
        self._network = None
    
    @staticmethod
    def _augment(data, crop_dim, color_jitter=True):
        """
        Augment a dataset by random cropping.
        The data must have shape (n_samples, n_channels, width, height).
        """

        new_data = []
        for img in data:
            new_img = np.array(img)

            if color_jitter:
                new_img = new_img.transpose(1, 2, 0)
                new_img = dataprocessing.color_jitter(new_img)
                new_img = new_img.transpose(2, 0, 1)
                new_img = keras.backend.cast_to_floatx(new_img) / 255.0

            new_img = random_crop(new_img, crop_dim)
        
            new_data.append(new_img)
        return np.array(new_data)

    @staticmethod
    def _iterate_minibatches(X, y, batch_size, shuffle=False):
        """
        Generates data batches to feed into the network.
        """

        if shuffle:
            X, y = sklearn.utils.shuffle(X, y)

        n_batches = X.shape[0] / batch_size
        for i in range(n_batches):
            idx = i * batch_size
            yield X[idx:idx+batch_size], y[idx:idx+batch_size]

    @staticmethod
    def _load_data(path, target_size):
        """Loads and preprocesses training and validation data."""

        # check if this is pickled data
        if os.path.isfile(path):
            (X_train, y_train), (X_val, y_val), class_map = DeepLearning._load_pickled_data(path)
        else:
            (X_train, y_train), (X_val, y_val), class_map = dataprocessing.split_data(
                                            path, target_size, transpose=True)

        n_classes = len(class_map)

        X_train = keras.backend.cast_to_floatx(X_train) / 255.0
        X_val = keras.backend.cast_to_floatx(X_val) / 255.0

        y_train = np_utils.to_categorical(y_train, n_classes)
        y_val = np_utils.to_categorical(y_val, n_classes)

        return (X_train, y_train), (X_val, y_val), n_classes

    @staticmethod
    def _load_pickled_data(path):
        """
        Loads pickled data. 
        Assume that the data are class_map, test set, train set.
        """

        with open(path, 'rb') as fp:
            class_map = cPickle.load(fp)
            test_data = cPickle.load(fp)
            train_data = cPickle.load(fp)

            return train_data, test_data, class_map


    def _train_data(self, train_data, n_classes, optimizer, val_data=None, n_epochs=100, 
                    batch_size=32, crop_dim=128, **train_params):
        """Helper function for training data."""

        X_train, y_train = train_data
        X_val, y_val = val_data

        # load and compile network
        input_shape = (3, crop_dim, crop_dim)
        self._network = network = self._model_module.build_model(
                                    input_shape, n_classes)

        opt = self.optimizers_list[optimizer](**train_params)

        network.compile(optimizer=opt, loss='categorical_crossentropy', 
                    metrics=['accuracy'])

        print ""
        print "Begin training ..."
        print "Optimizer: ", opt.__class__.__name__
        print "Parameters:", opt.get_config()

        best_val_acc = 0.0
        best_val_loss = None
        best_weights = None

        for epoch in range(1, n_epochs+1):
            # train on batches
            train_err = 0.0
            train_batches = 0
            start_time = time.time()

            for X_batch, y_batch in self._iterate_minibatches(X_train, y_train, 
                                                    batch_size, shuffle=True):
                X_augmented = self._augment(X_batch, crop_dim)
                
                batch_err, batch_acc = network.train_on_batch(X_augmented, 
                                                            y_batch)
                train_err += batch_err
                train_batches += 1

            # evaluate on validation data
            X_val_cropped = center_crop(X_val, crop_dim)
            val_err, val_acc = network.evaluate(X_val_cropped, y_val, 
                                                batch_size=batch_size)
            best_val_acc = max(val_acc, best_val_acc)
            if (best_val_loss is None) or (best_val_loss > val_err):
                best_val_loss = val_err
                best_weights = network.get_weights()

            # print out information at the end of each epoch
            print "Epoch {}/{} took {:.3f}s: ".format(epoch, n_epochs,
                                                time.time()-start_time),
            print "training loss = {} --".format(train_err / train_batches),
            print "validation loss = {} --".format(val_err),
            print "validation acc = {:.2f}%.".format(val_acc * 100)

        print "Best validation accuracy = {:.2f},".format(best_val_acc * 100),
        print "Best validation loss = {}".format(best_val_loss)

        # save model to use as a classifier
        network.set_weights(best_weights)
        network.save('experiments/my_model.h5')

        return best_val_loss


    def train(self, path, optimizer, n_epochs=100, batch_size=32, target_size=(140, 140),
                crop_dim=224, **train_params):
        """Train a classfier with the given model."""

        # split the data and determine components
        print "Loading data ..."

        (X_train, y_train), (X_val, y_val), n_classes = self._load_data(path,
                                                                    target_size)

        print ""
        print "Number of training samples:", X_train.shape, X_train.dtype
        print "Number of validation samples:", X_val.shape, X_val.dtype

        self._train_data((X_train, y_train), n_classes, optimizer, val_data=(X_val, y_val), 
                n_epochs=n_epochs, batch_size=batch_size, crop_dim=crop_dim, **train_params)


    def grid_search(self, dirname, space, max_evals=10, target_size=(140, 140), 
                    crop_dim=128):
        """
        Optimize parameters using grid search.
        The search space is expected to provide number of epochs, batch_size
        and the training algorithm (adagrad, adadelta, ...).
        """

        # split the data and determine components
        print "Splitting data ..."

        (X_train, y_train), (X_val, y_val), n_classes = self._load_data(dirname,
                                                                target_size)

        print ""
        print "Number of training samples:", X_train.shape, X_train.dtype
        print "Number of validation samples:", X_val.shape, X_val.dtype

        # define objective function to minimize
        def objective(params):
            batch_size = params['batch_size']
            n_epochs = params['num_epochs']
            
            optimizer = params['optimizer']['type']
            opt_params = params['optimizer']['params']

            score = self._train_data((X_train, y_train), n_classes, optimizer,
                val_data=(X_val, y_val), n_epochs=n_epochs, batch_size=batch_size, 
                crop_dim=crop_dim, **opt_params)

            sys.stdout.flush()
            return {'loss': score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, 
                    trials=trials)

        print 'best:', best