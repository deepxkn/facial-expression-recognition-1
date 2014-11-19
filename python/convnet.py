#NB: install latest theano with command
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.

This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time

import numpy as np
from pylearn2.expr.preprocessing import global_contrast_normalize
from PIL import Image

import theano
import theano.tensor as tensor
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
#theano.config.exception_verbosity='high'
#theano.config.optmizer='None'

from logistic_sgd import *
from mlp import HiddenLayer
from util import *


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = tensor.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.01, n_epochs=100,
                    nkerns=[20, 50], batch_size=100,
                    filter_size = (5, 5),
                    pool_size = (2, 2),
                    training_data=None, validation_data=None,
                    test_data=None, image_dim=32):
    """
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type training_data: tuple of (np.ndarry, np.ndarry)
    :param training_data: tuple of (input, target), where
        input is an np.ndarray of D dimensions (a matrix)
        whose rows correspond to an example. target is a
        np.ndarray of 1 dimensions (vector) that has length equal to
        the number of rows in the input. It should give the target
        value to the example with the same index in the input.

    :type validation_data: as above


    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    if training_data is None or validation_data is None:
        print "No dataset given."
        sys.exit(1)

    rng = np.random.RandomState(23455)

    # cast the data as tensor variables
    training_data, validation_data, test_data = \
        prepare_data(training_data, validation_data, test_data)

    train_set_x, train_set_y = training_data
    valid_set_x, valid_set_y = validation_data
    if test_data is not None:
        test_set_x, test_set_y = test_data

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(train_set_x.shape[0].eval())
    n_valid_batches = int(valid_set_x.shape[0].eval())
    if test_data is not None:
        n_test_batches = int(test_set_x.shape[0].eval())

    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    if test_data is not None:
        n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = tensor.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = tensor.matrix('x')   # data presented as rasterized images
    y = tensor.ivector('y')  # labels presented as 1D vector of [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    input_size = (image_dim, image_dim)

    # Reshape matrix of rasterized images of shape (batch_size, image_dim * image_dim)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, input_size[0], input_size[1]))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size
    # maxpooling reduces this further by a half
    # 4D output tensor is thus of shape (batch_size, nkerns[0],
    # new_image_dim, new_image_dim)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, input_size[0], input_size[1]),
        filter_shape=(nkerns[0], 1, filter_size[0], filter_size[1]),
        poolsize=pool_size
    )

    input_size = ((input_size[0] - filter_size[0] + 1) / pool_size[0],
                  (input_size[1] - filter_size[1] + 1) / pool_size[1])

    # Construct the second convolutional pooling layer
    # filtering reduces the new image size as before
    # maxpooling reduces this further by a half
    # 4D output tensor is thus of shape (nkerns[0], nkerns[1],
    # (((((image_dim-5+1)/2)-5+1)/2), (((image_dim-5+1)/2)-5+1)/2))
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], input_size[0], input_size[1]),
        filter_shape=(nkerns[1], nkerns[0], filter_size[0], filter_size[1]),
        poolsize=pool_size
    )

    input_size = ((input_size[0] - filter_size[0] + 1) / pool_size[0],
                  (input_size[1] - filter_size[1] + 1) / pool_size[1])

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * input_size[0] * input_size[1],
        n_out=500,
        activation=tensor.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=8)


    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to give predictions
    training_predictions = theano.function(
        inputs=[index],
        outputs=[layer3.y_pred],
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a function to compute the mistakes that are made by the model
    training_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    if test_data is not None:
        test_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = tensor.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size : (index + 1) * batch_size],
            y: train_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter

            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on training and validation sets
                training_losses = [
                    training_model(i)
                    for i in xrange(n_train_batches)
                ]
                this_training_loss = np.mean(training_losses)

                result = [training_predictions(i) for i in xrange(n_train_batches)]
                result = np.asarray(result);
                print result

                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, training error %f %%, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_training_loss * 100,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    if test_data is not None:
                        test_losses = [
                            test_model(i)
                            for i in xrange(n_test_batches)
                        ]
                        test_score = np.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

###############################################################################
# DEBUGGING
###############################################################################
# need to get the dimensions right for visualising the filters

        # Plot filters after each training epoch
        # Construct image from the weight matrix
        #image = Image.fromarray(
        #    tile_raster_images(
        #        X=layer1.W.get_value(borrow=True).T,
        #        img_shape=(image_dim, image_dim),
        #        tile_shape=(10, 10),
        #        tile_spacing=(1, 1)
        #    )
        #)

        #image.save('filters_at_epoch_%i.png' % epoch)
        #plotting_stop = time.clock()
        #plotting_time += (plotting_stop - plotting_start)

###############################################################################

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    labeled_training, labeled_training_labels = load_labeled_training(flatten=True)
    labeled_training -= np.mean(labeled_training)
    assert labeled_training.shape == (2925, 1024)

    from zca import ZCA
    zca = ZCA().fit(labeled_training)
    labeled_training = zca.transform(labeled_training)
    render_matrix(labeled_training[:100,:], flattened=True)

    #render_matrix(labeled_training[:100,:], flattened=True)
    #labeled_training = global_contrast_normalize(labeled_training, use_std=True)
    #render_matrix(labeled_training[:100,:], flattened=True)

    # dumb validation set partition for now
    shuffle_in_unison(labeled_training, labeled_training_labels)
    valid_split = labeled_training.shape[0] // 4
    train_data, train_labels = (labeled_training[valid_split:, :], labeled_training_labels[valid_split:])
    valid_data, valid_labels = (labeled_training[:valid_split, :], labeled_training_labels[:valid_split])

    evaluate_lenet5(
            training_data=(train_data, train_labels),
            validation_data=(valid_data, valid_labels)
            )

def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
