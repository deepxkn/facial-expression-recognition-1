# See README before reading this file
#
# This script creates a preprocessed version of a dataset using pylearn2.#
# It's not necessary to save preprocessed versions of your dataset to
# disk but this is an instructive example, because later we can show
# how to load your custom dataset in a yaml file.
#
# This is also a common use case because often you will want to preprocess
# your data once and then train several models on the preprocessed data.

from pylearn2.utils import serial
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.datasets import preprocessing
from pylearn2.format.target_format import convert_to_one_hot

import pylab as plt
import cPickle as pickle
import numpy as np

import util
import dictionary_learning

if __name__ == "__main__":

    train_data, train_labels = util.load_labeled_training(flatten=True, zero_index=True)
    train_data = util.standardize(train_data)
    test_data = util.load_public_test(flatten=True)
    test_data = util.standardize(test_data)

    train_data_20, _, test_data_20 = dictionary_learning.get_dictionary_data(n_comp=20, zero_index=True)
    train_data_100, _, test_data_100 = dictionary_learning.get_dictionary_data(n_comp=100, zero_index=True)

    # convert the training labels into one-hot format, as required by the pylearn2 model
    train_labels = convert_to_one_hot(train_labels, dtype='int64', max_labels=7, mode='stack')

    # pickle the data
    serial.save('training_data_for_pylearn2.pkl', train_data)
    serial.save('training_data_20_components_for_pylearn2.pkl', train_data_20)
    serial.save('training_data_100_components_for_pylearn2.pkl', train_data_100)

    serial.save('training_labels_for_pylearn2.pkl', train_labels)

    serial.save('test_data_for_pylearn2.pkl', test_data)
    serial.save('test_data_20_components_for_pylearn2.pkl', test_data_20)
    serial.save('test_data_100_components_for_pylearn2.pkl', test_data_100)

    ## test that pickling works
    #with open('training_data_for_pylearn2.pkl') as d:
    #    data_check = pickle.load(d)
    #with open('training_labels_for_pylearn2.pkl') as l:
    #    labels_check = pickle.load(l)
    #with open('preprocessed_test_for_pylearn2.pkl') as t:
    #    test_check = pickle.load(t)

    ## check y's
    #for index in range(len(labels_check)):
    #    for inner_index in range(len(labels_check[index])):
    #        assert labels_check[index, inner_index] == train_labels[index, inner_index]

    ## check x's
    #for index in range(len(data_check)):
    #    for inner_index in range(len(data_check[index])):
    #        assert data_check[index, inner_index] == train_data[index, inner_index]

    ## check test
    #for index in range(len(test_check)):
    #    for inner_index in range(len(test_check[index])):
    #        assert test_check[index, inner_index] == test_images[index, inner_index]
