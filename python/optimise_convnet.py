# wrapper for spearmint hyperparameter optimisation
from convnet import *
import util
import numpy as np
import pprint

labeled_training, labeled_training_labels = util.load_labeled_training(flatten=True)

from zca import ZCA
zca = ZCA().fit(labeled_training)
labeled_training = zca.transform(labeled_training)

# dumb validation set partition for now
util.shuffle_in_unison(labeled_training, labeled_training_labels)
valid_split = labeled_training.shape[0] // 4
train_data, train_labels = (labeled_training[valid_split:, :], labeled_training_labels[valid_split:])
valid_data, valid_labels = (labeled_training[:valid_split, :], labeled_training_labels[:valid_split])

def main(job_id, params):
    pprint.pprint(params)
    return evaluate_lenet5(
        learning_rate=params['learning rate'],
        learning_rate_decay=params['learning rate decay'],
        n_epochs=params['number of epochs'],
        patience=params['patience'],
        patience_increase=params['patience increase'],
        improvement_threshold=params['improvement threshold'],
        nkerns=[int(n) for n in params['number of kernels'].split()],
        batch_size=params['batch size'],
        filter_size = (params['filter size'], params['filter size'])
        pool_size = (params['pool size'], params['pool size'])
        n_convpool_layers = params['number of convpool layers'],
        n_hidden_layers = params['number of hidden layers'],
        n_hidden_units = params['number of hidden units'],
        convpool_layer_activation=params['convpool layer activation function'],
        hidden_layer_activation=params['hidden layer activation function'],
        training_data=(train_data, train_labels),
        validation_data=(valid_data, valid_labels)
    )
