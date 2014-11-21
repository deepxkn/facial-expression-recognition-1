from convnet import *
import util
import itertools
import numpy as np
import theano.tensor as tensor
import pprint

parameters = {
    # hyperparameters
    'number of epochs' :                    (1000,),
    'batch size' :                          (30, 50, 100),
    'filter size' :                         ((5, 5),),
    'pool size' :                           ((2, 2),),
    'learning rate' :                       (10**x for x in range(-4, 1)),
    'learning rate decay' :                 (None, 0.995),
    'patience' :                            (10000,),
    'patience increase' :                   (2,),
    'improvement threshold' :               (0.995,),
    # architecture
    'convpool layer activation function':   (tensor.tanh,),
    'hidden layer activation function':     (tensor.tanh,),
    'number of convpool layers' :           (2, 3),
    'number of hidden layers' :             (1, 2),
    'number of hidden units' :              (50, 100, 500, 1000)
    #'number of hidden units' :              (500, 1000)
}

experiment_conditions = [dict(zip(parameters, x)) for x in itertools.product(*parameters.values())]

labeled_training, labeled_training_labels = util.load_labeled_training(flatten=True)
labeled_training -= np.mean(labeled_training)

from zca import ZCA
zca = ZCA().fit(labeled_training)
labeled_training = zca.transform(labeled_training)
#render_matrix(labeled_training[:100,:], flattened=True)

#render_matrix(labeled_training[:100,:], flattened=True)
#labeled_training = global_contrast_normalize(labeled_training, use_std=True)
#render_matrix(labeled_training[:100,:], flattened=True)

# dumb validation set partition for now
util.shuffle_in_unison(labeled_training, labeled_training_labels)
valid_split = labeled_training.shape[0] // 4
train_data, train_labels = (labeled_training[valid_split:, :], labeled_training_labels[valid_split:])
valid_data, valid_labels = (labeled_training[:valid_split, :], labeled_training_labels[:valid_split])

f = open('out.txt', 'w')


for i in range(len(experiment_conditions)):
    cond = experiment_conditions[i]
    print '==================================================================='
    sys.stderr.write('condition number ' + str(i) + ' out of ' + str(len(experiment_conditions)) + '\n')
    sys.stderr.write(str(cond) + '\n')
    f.write('============================================================')
    pprint.pprint(cond)
    f.write(str(cond))
    f.write('\n')
    loss = evaluate_lenet5(
        learning_rate=cond['learning rate'],
        n_epochs=cond['number of epochs'],
        patience=cond['patience'],
        patience_increase=cond['patience increase'],
        improvement_threshold=cond['improvement threshold'],
        nkerns=[20, 50, 100],
        batch_size=cond['batch size'],
        filter_size =cond['filter size'],
        pool_size = cond['pool size'],
        n_convpool_layers = cond['number of convpool layers'],
        n_hidden_layers = cond['number of hidden layers'],
        n_hidden_units = cond['number of hidden units'],
        convpool_layer_activation=tensor.tanh,
        hidden_layer_activation=tensor.tanh,
        training_data=(train_data, train_labels),
        validation_data=(valid_data, valid_labels)
    )
    f.write('best validation loss: ' + str(loss))
    sys.stderr.write('best validation loss: ' + str(loss) + '\n')
    f.write('\n')
