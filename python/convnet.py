#NB: install latest theano with command
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
import os
import sys
import time

import itertools
import numpy as np
import operator
from PIL import Image

import theano
import theano.tensor as tensor
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
#theano.config.exception_verbosity='high'
theano.config.optmizer='fast_compile'

import classifiers
# set the package name to allow relative imports
if __name__ == "__main__" and __package__ is None:
    __package__ = "classifiers.convnet"
from .. import util


class ConvPoolLayer(object):
    """Pooling layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),
            activation=tensor.tanh):
        """
        Allocate a ConvPoolLayer with shared variable internal parameters.

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
        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=tensor.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = tensor.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # parameters of the model
        self.params = [self.W, self.b]

class HiddenLayerWithDropout(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
        activation, dropout_rate, W=None, b=None):

        super(HiddenLayerWithDropout, self).__init__(
            rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
            activation=activation
        )

        self.output = dropout_from_layer(rng, self.output, p=dropout_rate)

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        if W is None:
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
            self.W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        else:
            self.W = W

        if b is None:
            # initialize the biases b as a vector of n_out 0s
            self.b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            self.b = b

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plane-k
        self.p_y_given_x = tensor.nnet.softmax(tensor.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = tensor.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def test(self, x):
        return tensor.argmax(tensor.nnet.softmax(tensor.dot(x, self.W) + self.b), axis=1)

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # tensor.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] tensor.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[tensor.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and tensor.mean(LP[tensor.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -tensor.mean(tensor.log(self.p_y_given_x)[tensor.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the tensor.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return tensor.mean(tensor.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

    # 1 - p is the probability of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)

    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * tensor.cast(mask, theano.config.floatX)

    return output

def prepare_data(training, validation, test=None):
    ''' Prepares the dataset to feed into the model
    '''
    train_data, train_labels = training
    valid_data, valid_labels = validation
    if test is not None:
        test_data = test

    # check all dimensions
    assert train_data.shape[0] == train_labels.shape[0]
    assert valid_data.shape[0] == valid_labels.shape[0]

    assert train_data.shape[1] == 32*32
    assert valid_data.shape[1] == 32*32
    if test is not None:
        assert test_data.shape[1] == 32*32

    assert len(train_labels.shape) == 1
    assert len(valid_labels.shape) == 1

    train_set, valid_set = (train_data, train_labels), \
            (valid_data, valid_labels)
    if test is not None:
        # create a dummy label matrix
        test_labels = np.zeros((test_data.shape[0],), dtype=theano.config.floatX)
        test_set = (test_data, test_labels)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, tensor.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    if test is not None:
        test_set_x, test_set_y = shared_dataset(test_set)

    if test is not None:
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), None]

    return rval

def relu(x):
    return theano.tensor.switch(x<0, 0, x)

def evaluate_lenet5(initial_learning_rate=0.08,
                    learning_rate_decay=0.998,
                    n_epochs=100,
                    patience=10000,
                    patience_increase=2,
                    improvement_threshold=0.995,
                    nkerns=[30, 50],
                    batch_size=100,
                    filter_size = (3, 3),
                    pool_size = (2, 2),
                    n_convpool_layers = 1,
                    n_hidden_layers = 1,
                    hidden_layer_sizes = [100, 100],
                    convpool_layer_activation=tensor.tanh,
                    hidden_layer_activation=relu,
                    dropout=True,
                    dropout_rates = [ 0.2, 0.5, 0.5 ],
                    training_data=None,
                    validation_data=None,
                    test_data=None,
                    image_dim=32):
    """
    :type initial learning_rate: float
    :param initial learning_rate: learning rate used (factor for the stochastic
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

    # early-stopping parameters
    patience
    # look as this many examples regardless
    patience_increase
    # wait this much longer when a new best is
                           # found
    improvement_threshold
    # a relative improvement of this much is
                                   # considered significant
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

    x = tensor.matrix('x')   # data presented as rasterized images
    y = tensor.ivector('y')  # labels presented as 1D vector of [int] labels

    learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    input_size = (image_dim, image_dim)
    # Reshape matrix of rasterized images of shape (batch_size, image_dim * image_dim)
    # to a 4D tensor, compatible with our ConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, input_size[0], input_size[1]))

    conv_pool_layers = []

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size
    # maxpooling reduces this further by a half
    # 4D output tensor is thus of shape (batch_size, nkerns[0],
    # new_image_dim, new_image_dim)
    conv_pool_layers.append(ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, input_size[0], input_size[1]),
        filter_shape=(nkerns[0], 1, filter_size[0], filter_size[1]),
        poolsize=pool_size,
        activation=convpool_layer_activation
    ))

    input_size = ((input_size[0] - filter_size[0] + 1) / pool_size[0],
                  (input_size[1] - filter_size[1] + 1) / pool_size[1])

    # Construct the next convolutional pooling layers
    for layer_num in range(1, n_convpool_layers):
        conv_pool_layers.append(ConvPoolLayer(
            rng,
            input=conv_pool_layers[layer_num-1].output,
            image_shape=(batch_size, nkerns[layer_num-1], input_size[0], input_size[1]),
            filter_shape=(nkerns[layer_num], nkerns[layer_num-1], filter_size[0], filter_size[1]),
            poolsize=pool_size,
            activation=convpool_layer_activation
        ))

        input_size = ((input_size[0] - filter_size[0] + 1) / pool_size[0],
                      (input_size[1] - filter_size[1] + 1) / pool_size[1])

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    # construct a fully-connected sigmoidal layer
    hidden_layers = []
    hidden_layers_with_dropout = []
    hidden_layer_weight_matrix_sizes = zip(hidden_layer_sizes, hidden_layer_sizes[1:])

    next_layer_input = conv_pool_layers[-1].output.flatten(2)
    next_dropout_layer_input = dropout_from_layer(rng, conv_pool_layers[-1].output.flatten(2), p=dropout_rates[0])

    layer_counter = 0

    for n_in, n_out in hidden_layer_weight_matrix_sizes[:-1]:
        if layer_counter == 0:
            n_in = nkerns[n_convpool_layers-1] * input_size[0] * input_size[1],
        next_dropout_layer = HiddenLayerWithDropout(
            rng=rng,
            input=next_dropout_layer_input,
            n_in=n_in, n_out=n_out,
            dropout_rate=dropout_rates[layer_counter + 1],
            activation=hidden_layer_activation
        )

        hidden_layers_with_dropout.append(next_dropout_layer)
        next_dropout_layer_input = next_dropout_layer.output

        # Reuse the paramters from the dropout layer here, in a different
        # path through the graph.
        next_layer = HiddenLayer(
            rng=rng,
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
            b=next_dropout_layer.b,
            n_in=n_in, n_out=n_out,
            activation=hidden_layer_activation
        )

        hidden_layers.append(next_layer)
        next_layer_input = next_layer.output

        layer_counter += 1

    # Set up the output layer
    n_in, n_out = hidden_layer_weight_matrix_sizes[-1]
    dropout_output_layer = LogisticRegression(
        input=next_dropout_layer_input,
        n_in=n_in, n_out=8
    )

    # Again, reuse paramters in the dropout output.
    output_layer = LogisticRegression(
        input=next_layer_input,
        # scale the weight matrix W with (1-p)
        W=dropout_output_layer.W * (1 - dropout_rates[-1]),
        b=dropout_output_layer.b,
        n_in=n_in, n_out=8
    )

    # Use the negative log likelihood of the logistic regression layer as
    # the objective.
    dropout_cost = dropout_output_layer.negative_log_likelihood(y)
    cost = output_layer.negative_log_likelihood(y)

    # create a function to give predictions
    training_predictions = theano.function(
        inputs=[],
        outputs=[output_layer.test(x)],
        givens={
            x: train_set_x
        }
    )

    validation_predictions = theano.function(
        inputs=[],
        outputs=[output_layer.test(x)],
        givens={
            x: valid_set_x
        }
    )

    if test_data is not None:
        test_predictions = theano.function(
            inputs=[index],
            outputs=[output_layer.y_pred],
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
            }
        )

    # create a function to compute the mistakes that are made by the model
    training_model = theano.function(
        [index],
        output_layer.errors(y),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        output_layer.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = [layer.params for layer in conv_pool_layers] + \
             [layer.params for layer in hidden_layers] + \
             [ output_layer.params ]
    params = params[::-1] # reverse the array
    params = reduce(operator.add, params) # flatten the array

    # create a list of gradients for all model parameters
    if dropout is True:
        grads = tensor.grad(dropout_cost, params)
    else:
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

    output = dropout_cost if dropout else cost
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size : (index + 1) * batch_size],
            y: train_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    # function for decaying the learning rate after each epoch
    decay_learning_rate = theano.function(
        inputs=[],
        updates={
            learning_rate: learning_rate * learning_rate_decay
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

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

                validation_losses = [
                    validate_model(i) for i
                    in xrange(n_valid_batches)
                ]
                this_validation_loss = np.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, learning_rate %f, training error %f %%, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       learning_rate.get_value(borrow=True),
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

            if patience <= iter:
                done_looping = True
                break

        decay_learning_rate()

    if test_data is not None:
        test_pred = [
            test_predictions(i) for i
            in xrange(n_test_batches)
        ]
        test_pred = list(itertools.chain.from_iterable(test_pred))
        test_pred = list(itertools.chain.from_iterable(test_pred))

        return best_validation_loss, test_pred

    else:
        return best_validation_loss, None

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

    labeled_training, labeled_training_labels = util.load_labeled_training(flatten=True)
    #labeled_training -= np.mean(labeled_training)
    assert labeled_training.shape == (2925, 1024)

    labeled_training = util.standardize(labeled_training)
    #util.render_matrix(labeled_training[:100,:], flattened=True)

    test_images = util.load_public_test(flatten=True)
    test_images = util.standardize(test_images)

    #from zca import ZCA
    #zca = ZCA().fit(labeled_training)
    #labeled_training = zca.transform(labeled_training)
    #render_matrix(labeled_training[:100,:], flattened=True)

    #render_matrix(labeled_training[:100,:], flattened=True)
    #labeled_training = global_contrast_normalize(labeled_training, use_std=True)
    #render_matrix(labeled_training[:100,:], flattened=True)

    # dumb validation set partition for now
    util.shuffle_in_unison(labeled_training, labeled_training_labels)
    valid_split = labeled_training.shape[0] // 12
    train_data, train_labels = (labeled_training[valid_split:, :], labeled_training_labels[valid_split:])
    valid_data, valid_labels = (labeled_training[:valid_split, :], labeled_training_labels[:valid_split])

    _, test_labels = evaluate_lenet5(
            training_data=(train_data, train_labels),
            validation_data=(valid_data, valid_labels),
            test_data=test_images,
            filter_size=(3, 3)
            )

    print test_labels
    util.write_results(test_labels, 'predictions.csv')

