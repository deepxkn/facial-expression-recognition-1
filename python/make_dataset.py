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
from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import Grad_M_Step

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import DictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

import pylab as plt
import cPickle as pickle
import numpy as np

import util

if __name__ == "__main__":
    # load the training data
    train_data, train_labels = util.load_labeled_training(flatten=True, zero_index=True)
    unlabeled_training = util.load_unlabeled_training(flatten=True)

    # load the test data
    test_images = util.load_public_test(flatten=True)
    # preprocess the test data as is done to the training data in the yaml file
    test_images = util.standardize(test_images)

    # convert the training labels into one-hot format, as required by the pylearn2 model
    train_labels = convert_to_one_hot(train_labels, dtype='int64', max_labels=7, mode='stack')

    ###########################################################################
    # pickling preprocessed unlabeled data
    # don't modify this without modifying the retraining a new model with the new data

    #unlabeled_training -= unlabeled_training.mean(axis=1) # centralise the data
    #unlabeled_training /= numpy.sqrt(unlabeled_training.var(axis=1, ddof=1)) # normalise the data

    #nolabelmatrix =  DenseDesignMatrix(X=unlabeled_training)
    #zca = ZCA()
    #zca.fit(nolabelmatrix)

<<<<<<< HEAD
    print type(unlabeled_training)
    code = DictionaryLearning(n_components=100, max_iter=100)
    code.fit(unlabeled_training)
    new = code.transform(train_data[:20])
    util.render_matrix(train_data[:20], flattened=True)
=======
    #serial.save('zca_fit_with_unlabeled_training.pkl', zca)
    ###########################################################################

    # dictionary learning
    #print type(unlabeled_training)
    #code = DictionaryLearning(n_components=100)
    #code.fit(unlabeled_training)
    #code.transform(train_data)
    #util.render_matrix(train_data[:20], flattened=True)
>>>>>>> d57ef3b813fb9ea300e6bdf5e80244a15df0cfad

    # create the spike-and-slab encoding dictionary
    #unlabeled_training = DenseDesignMatrix(X=unlabeled_training)
    #m, D = unlabeled_training.X.shape # number of visible units
    #N = 300 # number of hidden units

    #s3c = S3C(nvis = D,
    #    nhid = N,
    #    irange = .1,
    #    init_bias_hid = 0.,
    #    init_B = 3.,
    #    min_B = 1e-8,
    #    max_B = 1000.,
    #    init_alpha = 1., min_alpha = 1e-8, max_alpha = 1000.,
    #    init_mu = 1., e_step = None,
    #    m_step = Grad_M_Step(),
    #    min_bias_hid = -1e30, max_bias_hid = 1e30,
    #)

    #s3c.make_pseudoparams()
    #s3c.learn(unlabeled_training, m)

    # pickle the data
    serial.save('training_data_for_pylearn2.pkl', train_data)
    serial.save('training_labels_for_pylearn2.pkl', train_labels)
    serial.save('preprocessed_test_for_pylearn2.pkl', test_images)

    # test that pickling works
    with open('training_data_for_pylearn2.pkl') as d:
        data_check = pickle.load(d)
    with open('training_labels_for_pylearn2.pkl') as l:
        labels_check = pickle.load(l)
    with open('preprocessed_test_for_pylearn2.pkl') as t:
        test_check = pickle.load(t)

    # check y's
    for index in range(len(labels_check)):
        for inner_index in range(len(labels_check[index])):
            assert labels_check[index, inner_index] == train_labels[index, inner_index]

    # check x's
    for index in range(len(data_check)):
        for inner_index in range(len(data_check[index])):
            assert data_check[index, inner_index] == train_data[index, inner_index]

    # check test
    for index in range(len(test_check)):
        for inner_index in range(len(test_check[index])):
            assert test_check[index, inner_index] == test_images[index, inner_index]
