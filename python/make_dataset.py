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

from sklearn.cross_validation import StratifiedKFold

import cPickle as pickle

import util

if __name__ == "__main__":
    n_folds = 4

    view_converter = DefaultViewConverter((32, 32, 1), ('b', 0, 1, 'c'))
    #unlabeled_training = util.load_unlabeled_training(flatten=True)
    test_images = util.load_public_test(flatten=True)
    #unlabeled_training_matrix = DenseDesignMatrix(X=unlabeled_training, view_converter=view_converter)
    test_matrix = DenseDesignMatrix(X=test_images, view_converter=view_converter)

    labeled_training, labeled_training_labels = util.load_labeled_training(flatten=True, zero_index=True)

    skf = StratifiedKFold(labeled_training_labels, n_folds)

    for i, (train_index, valid_index) in enumerate(skf):
        #print train_index, valid_index
        train_data = labeled_training[train_index]
        train_labels = labeled_training_labels[train_index]

        valid_data = labeled_training[valid_index]
        valid_labels = labeled_training_labels[valid_index]

        #print train_labels[:20]
        #util.render_matrix(train_data[:20], flattened=True)

        # convert the training labels into one-hot format, as required by the pylearn2 model
        train_labels = convert_to_one_hot(train_labels, dtype='int64', max_labels=7, mode='stack')
        valid_labels = convert_to_one_hot(valid_labels, dtype='int64', max_labels=7, mode='stack')

        #print train_labels[:20]
        #util.render_matrix(train_data[:20], flattened=True)
        #print valid_labels[:20]
        #util.render_matrix(valid_data[:20], flattened=True)

        labeled_training_matrix = DenseDesignMatrix(X=train_data, y=train_labels, view_converter=view_converter, y_labels=7)
        labeled_validation_matrix = DenseDesignMatrix(X=valid_data, y=valid_labels, view_converter=view_converter, y_labels=7)

        # First we want to pull out small patches of the images, since it's easier
        # to train an RBM on these
        #pipeline.items.append(
            #preprocessing.ExtractPatches(patch_shape=(8, 8), num_patches=150000)
        #)

        # global contrast normalisation
        gcn = preprocessing.GlobalContrastNormalization(subtract_mean=True, sqrt_bias=0.0, use_std=True)
        labeled_training_matrix.apply_preprocessor(gcn)
        labeled_validation_matrix.apply_preprocessor(gcn)

        # ZCA whitening
        #zca = preprocessing.ZCA()
        #zca.fit(unlabeled_training_matrix.get_design_matrix())
        #labeled_training_matrix.apply_preprocessor(zca, can_fit=True) # can_fit=True means can fit whitening matrix to this dataset
        #test_matrix.apply_preprocessor(zca, can_fit=False)

        # Finally we save the dataset to the filesystem. We instruct the dataset to
        # store its design matrix as a numpy file because this uses less memory
        # when re-loading (Pickle files, in general, use double their actual size
        # in the process of being re-loaded into a running process).
        # The dataset object itself is stored as a pickle file.
        labeled_training_matrix.use_design_loc('train_design.npy')
        labeled_validation_matrix.use_design_loc('train_design.npy')

        #print labeled_training_matrix.y[500:520]
        #util.render_matrix(labeled_training_matrix.X[500:520], flattened=True)
        #print valid_labels[:20]
        #util.render_matrix(valid_data[:20], flattened=True)

        print type(labeled_training_matrix.X)
        with open('preprocessed_labeled_training_for_pylearn2_fold_{0}.pkl'.format(str(i)), 'wb') as f:
            pickle.dump(labeled_training_matrix, f)
        #serial.save('preprocessed_labeled_training_for_pylearn2_fold_{0}.pkl'.format(str(i)), labeled_training_matrix)
        serial.save('preprocessed_labeled_validation_for_pylearn2_fold_{0}.pkl'.format(str(i)), labeled_validation_matrix)

        with open('preprocessed_labeled_training_for_pylearn2_fold_{0}.pkl'.format(str(i))) as t:
            train_check = pickle.load(t)
        with open('preprocessed_labeled_validation_for_pylearn2_fold_{0}.pkl'.format(str(i))) as v:
            val_check = pickle.load(v)

        # check y's
        for index in range(len(train_check.y)):
            for inner_index in range(len(train_check.y[index])):
                assert train_check.y[index, inner_index] == labeled_training_matrix.y[index, inner_index]


        LHS = []
        for inner_list in train_check.X:
            LHS.append(str(inner_list))
        RHS = []
        for inner_list in labeled_training_matrix.X:
            RHS.append(str(inner_list))

        print set(LHS)
        print set(RHS)
        print len(set(LHS))

        assert set(LHS) == set(RHS)

        # check x's
        for index in range(len(train_check.X)):
            for inner_index in range(len(train_check.X[index])):
                assert train_check.X[index, inner_index] == labeled_training_matrix.X[index, inner_index]

    test_matrix.use_design_loc('train_design.npy')
    serial.save('preprocessed_test_for_pylearn2.pkl', test_matrix)
