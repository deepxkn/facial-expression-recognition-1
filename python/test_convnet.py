"""
Test that a smaller version of convolutional_network.ipynb works.

The differences (needed for speed) are:
    * output_channels: 4 instead of 64
    * train.stop: 500 instead of 50000
    * valid.stop: 50100 instead of 60000
    * test.start: 0 instead of non-specified
    * test.stop: 100 instead of non-specified
    * termination_criterion.max_epochs: 1 instead of 500

This should make the test run in about one minute.
"""

import os

from pylearn2.testing import skip
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse
#from pylearn2.scripts.jobman.experiment import train_experiment

#from jobman.tools import DD

def results_extractor(train_obj):
    channels = train_obj.model.monitor.channels
    test_y_misclass = channels['test_y_misclass'].val_record[-1]

    return DD(test_y_misclass=test_y_misclass)

def test_convolutional_network():

    skip.skip_if_no_data()

    yaml = open("conv.yaml", 'r').read()

    hyper_params = {
                    'learning_rate': 0.1,
                    'batch_size': 100,
                    'output_channels_conv1': 30,
                    'output_channels_conv2': 50,
                    'output_channels_conv3': 100,
                    'max_epochs': 100,
                    'num_hiddens_h1' : 50,
                    'num_hiddens_h2' : 100,
                    }
    yaml = yaml % (hyper_params)
    train = yaml_parse.load(yaml)
    train.main_loop()

if __name__ == '__main__':
    test_convolutional_network()
