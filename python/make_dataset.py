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

import cPickle as pickle

import util

if __name__ == "__main__":
    test_images = util.load_public_test(flatten=True)

    labeled_training, labeled_training_labels = util.load_labeled_training(flatten=True, zero_index=True)

    train_data = labeled_training
    train_labels = labeled_training_labels

    #print train_labels[:20]
    #util.render_matrix(train_data[:20], flattened=True)

    # convert the training labels into one-hot format, as required by the pylearn2 model
    train_labels = convert_to_one_hot(train_labels, dtype='int64', max_labels=7, mode='stack')

    serial.save('training_data_for_pylearn2.pkl', train_data)
    serial.save('training_labels_for_pylearn2.pkl', train_labels)

    with open('training_data_for_pylearn2.pkl') as d:
        data_check = pickle.load(d)
    with open('training_labels_for_pylearn2.pkl') as l:
        labels_check = pickle.load(l)

    # check y's
    for index in range(len(labels_check)):
        for inner_index in range(len(labels_check[index])):
            assert labels_check[index, inner_index] == train_labels[index, inner_index]

    # check x's
    for index in range(len(data_check)):
        for inner_index in range(len(data_check[index])):
            assert data_check[index, inner_index] == train_data[index, inner_index]

