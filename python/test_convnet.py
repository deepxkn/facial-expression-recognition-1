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

from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse
#from pylearn2.scripts.jobman.experiment import train_experiment

#from jobman.tools import DD

def results_extractor(train_obj):
    channels = train_obj.model.monitor.channels
    test_y_misclass = channels['test_y_misclass'].val_record[-1]
    train_y_misclass = channels['train_y_misclass'].val_record[-1]
    train_y_nll = channels['train_y_nll'].val_record[-1]
    test_y_nll = channels['test_y_nll'].val_record[-1]

    return DD(
        test_y_misclass=test_y_misclass,
        train_y_misclass=train_y_misclass,
        train_y_nll=train_y_nll,
        test_y_nll=test_y_nll
    )

def test_convolutional_network():
    yaml = open("conv_reproduce_results.yaml", 'r').read()

    hyper_params = {
                    'learning_rate': 0.2,
                    'batch_size': 40,
                    'output_channels_conv1': 70,
                    'output_channels_conv2': 80,
                    'output_channels_conv3': 60,
                    'max_epochs': 100,
                    'num_hiddens_h1' : 730,
                    'num_hiddens_h2' : 950,
                    }
    yaml = yaml % (hyper_params)
    train = yaml_parse.load(yaml)
    train.main_loop()

if __name__ == '__main__':
    test_convolutional_network()
