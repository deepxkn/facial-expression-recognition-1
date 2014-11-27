from theano.compat.six.moves import cPickle
import os
from pylearn2.testing.skip import skip_if_no_matplotlib
from pylearn2.models.mlp import MLP, Linear
from pylearn2.scripts.summarize_model import summarize

def summarize_model():

    raw_input()
    skip_if_no_matplotlib()
    summarize('convnet.pkl')
    show_weights('convnet.pkl', rescale='individual', border=True, out='first_layer_weights.png')

if __name__ == '__main__':
    summarize_model()
