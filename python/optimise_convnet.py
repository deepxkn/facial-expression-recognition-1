from convnet import *
#from util import load_labeled_training, shuffle_in_unison
# had to move the functions into this file because the name util was confounded
import numpy as np
import scipy.io
import pprint
from zca import ZCA
from collections import OrderedDict

def load_labeled_training(flatten=False):
    labeled = scipy.io.loadmat('../labeled_images.mat')
    labels = labeled['tr_labels']
    labels = np.asarray([l[0] for l in labels])
    images = labeled['tr_images']

    # permute dimensions so that the number of instances is first
    x, y, n = images.shape
    images = np.transpose(images, [2, 0, 1])
    assert images.shape == (n, x, y)

    # flatten the pixel dimensions
    if flatten is True:
        n, x, y = images.shape
        images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
        assert images.shape == (n, x*y)

    return images, labels

def shuffle_in_unison(a, b):
    """Shuffle two arrays in unison, so that previously aligned indices
    remain aligned. The arrays can be multidimensional; in any
    case, the first dimension is shuffled.
    """
    assert len(a) == len(b)
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def standardize(images):
    images = images.astype(float)
    mean = np.mean(images,axis=1)
    sd = np.sqrt(np.var(images, axis=1) + 1e-20)
    for i in range(images.shape[0]):
        for j in range(len(images[i])):
            images[i][j] -= mean[i]

    for i in range(images.shape[0]):
        for j in range(len(images[i])):
            images[i][j] /= sd[i]
    return images


# create phony parameters for debugging
test_params = OrderedDict({
    'number of epochs' : 100,
    'batch size' : 100,
    'filter size' : 3,
    'pool size' : 2,
    'learning rate' : 0.08,
    'learning rate decay' : 0.998,
    'convpool layer activation function' : 'tanh',
    'hidden layer activation function' : 'tanh',
    'number of convpool layers' : 1,
    'number of hidden layers' : 1,
    'number of hidden units' : 100,
    'product of number of kernels and number of pixel positions' : 60000,
    'standardisation' : 'True',
    'ZCA whitening' : 'True',
    'global contrast normalisation' : 'True'
    })

def main(job_id, params):
    labeled_training, labeled_training_labels = load_labeled_training(flatten=True)

    if params['ZCA whitening'] == 'True':
        zca = ZCA().fit(labeled_training)
        labeled_training = zca.transform(labeled_training)

    if params['global contrast normalisation'] == 'True':
        labeled_training -= np.mean(labeled_training)

    if params['standardisation'] == 'True':
        labeled_training -= standardize(labeled_training)

    # dumb validation set partition for now
    shuffle_in_unison(labeled_training, labeled_training_labels)
    valid_split = labeled_training.shape[0] // 4
    train_data, train_labels = (labeled_training[valid_split:, :], labeled_training_labels[valid_split:])
    valid_data, valid_labels = (labeled_training[:valid_split, :], labeled_training_labels[:valid_split])

    pprint.pprint(params)

    return evaluate_convnet(
        initial_learning_rate=params['learning rate'][0],
        learning_rate_decay=params['learning rate decay'][0],
        kernel_position_product=params['product of number of kernels and number of pixel positions'][0],
        n_epochs=params['number of epochs'][0],
        batch_size=params['batch size'][0],
        filter_size = (params['filter size'][0], params['filter size'][0]),
        pool_size = (params['pool size'][0], params['pool size'][0]),
        n_convpool_layers = params['number of convpool layers'][0],
        n_hidden_layers = params['number of hidden layers'][0],
        n_hidden_units = params['number of hidden units'][0],
        convpool_layer_activation=params['convpool layer activation function'][0],
        hidden_layer_activation=params['hidden layer activation function'][0],
        input_dropout=params['input layer dropout probability'][0],
        convpool_dropout=params['convpool layer dropout probability'][0],
        hidden_dropout=params['hidden layer dropout probability'][0],
        training_data=(train_data, train_labels),
        validation_data=(valid_data, valid_labels)
    )

if __name__ == '__main__':
    main(1601, test_params)
