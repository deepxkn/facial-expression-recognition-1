from __future__ import print_function

import os
import sys
import util
from theano.compat.six.moves import xrange
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.datasets.preprocessing import GlobalContrastNormalization
from pylearn2.utils import serial
from theano import tensor as T
from theano import function
import cPickle as pkl
import numpy as np
from scipy.stats import mode
import gzip

def make_majority_vote():

    model_paths = ['convnet_' + str(i+1) + '.pkl' for i in range(10)]
    out_path = 'submission.csv'

    models = []

    for model_path in model_paths:
        print('Loading ' + model_path + '...')
        try:
            with open(model_path, 'rb') as f:
                models.append(pkl.load(f))
        except Exception as e:
            try:
                with gzip.open(model_path, 'rb') as f:
                    models.append(pkl.load(f))
            except Exception as e:
                usage()
                print(model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: ")
                print(e)

    # load the test set
    with open('test_data_for_pylearn2.pkl', 'rb') as f:
        dataset = pkl.load(f)

    dataset = DenseDesignMatrix(X=dataset, view_converter=DefaultViewConverter(shape=[32, 32, 1], axes=['b', 0, 1, 'c']))
    preprocessor = GlobalContrastNormalization(subtract_mean=True, sqrt_bias=0.0, use_std=True)
    preprocessor.apply(dataset)

    predictions = []
    print('Model description:')
    print('')
    print(models[1])
    print('')

    for model in models:

        model.set_batch_size(dataset.X.shape[0])

        X = model.get_input_space().make_batch_theano()
        Y = model.fprop(X) # forward prop the test data

        y = T.argmax(Y, axis=1)

        f = function([X], y)

        x_arg = dataset.get_topological_view()
        y = f(x_arg.astype(X.dtype))

        assert y.ndim == 1
        assert y.shape[0] == dataset.X.shape[0]

        # add one to the results!
        y += 1

        predictions.append(y)

    predictions = np.array(predictions, dtype='int32')

    y = mode(predictions.T, axis=1)[0]
    y = np.array(y, dtype='int32')

    import itertools
    y = list(itertools.chain(*y))

    assert len(y) == dataset.X.shape[0]

    util.write_results(y, out_path)

    print('Wrote predictions to submission.csv.')
    return np.reshape(y, (1, -1))

if __name__ == '__main__':
    make_majority_vote()

