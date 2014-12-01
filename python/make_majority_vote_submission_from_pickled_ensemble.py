from __future__ import print_function

import os
import sys
import util
from theano.compat.six.moves import xrange
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.utils import serial
from theano import tensor as T
from theano import function
import cPickle as pkl
import numpy as np
from scipy.stats import mode


def usage():
    print("""usage: python make_submission.py model.pkl submission.csv)
Where model.pkl contains a list of trained pylearn2.models.mlp.MLP objects.
The script will make submission.csv, which you may then upload to the
kaggle site.""")


if len(sys.argv) != 3:
    usage()
    print("(You used the wrong # of arguments)")
    quit(-1)

_, model_path, out_path = sys.argv

if os.path.exists(out_path):
    usage()
    print(out_path+" already exists, and I don't want to overwrite anything just to be safe.")
    quit(-1)


models = []
try:
    for model in serial.load(model_path):
        models.append(model)
except Exception as e:
    usage()
    print(model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: ")
    print(e)

# load the test set
with open('preprocessed_test_for_pylearn2.pkl') as f:
    dataset = pkl.load(f)
dataset = DenseDesignMatrix(X=dataset, view_converter=DefaultViewConverter(shape=[32, 32, 1], axes=['b', 0, 1, 'c']))

print(models)
predictions = []
print(len(models))

for model in models:

    print(model)

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
    print(y)

predictions = np.array(predictions, dtype='int32')

y = mode(predictions.T, axis=1)[0]
y = np.array(y, dtype='int32')

import itertools
y = list(itertools.chain(*y))

assert len(y) == dataset.X.shape[0]
print(type(y[0]))
print(type(y))

util.write_results(y, out_path)
