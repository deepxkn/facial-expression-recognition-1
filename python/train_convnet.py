import os

from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse

def train_convolutional_network():
    yaml = open("conv_small_filters.yaml", 'r').read()

    hyper_params = {
                    'learning_rate': 0.2,
                    'init_momentum': 0.5,
                    'batch_size': 40,
                    'output_channels_conv1': 20,
                    'output_channels_conv2': 50,
                    'output_channels_conv3': 80,
                    'max_epochs': 100,
                    'num_hiddens_h1' : 256,
                    }
    yaml = yaml % (hyper_params)
    train = yaml_parse.load(yaml)
    train.main_loop()

if __name__ == '__main__': train_convolutional_network()
