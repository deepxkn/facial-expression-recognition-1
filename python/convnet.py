#NB: install latest theano with command
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
import os
import sys
import time

import itertools
import numpy as np
import operator
from PIL import Image
from collections import OrderedDict

import theano
import theano.tensor as tensor
from theano.ifelse import ifelse
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
#theano.config.exception_verbosity='high'
#theano.config.optmizer='fast_compile'

import util


class ConvPoolLayer(object):
    """Pooling layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),
            activation=tensor.tanh, use_bias=True, W=None, b=None):
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
        self.activation = activation

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
        if W is None:
            W = theano.shared(
                np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                name='W', borrow=True
            )

        if b is None:
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

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
        self.output = self.activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

class DropoutConvPoolLayer(ConvPoolLayer):
    def __init__(self, rng, input, filter_shape, image_shape, dropout_rate, poolsize=(2, 2),
                 activation=tensor.tanh, use_bias=True, W=None, b=None):

        super(DropoutConvPoolLayer, self).__init__(
                rng=rng, input=input, filter_shape=filter_shape, image_shape=image_shape,
                poolsize=poolsize, W=W, b=b,
                activation=activation, use_bias=use_bias)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation=tensor.tanh,
                 W=None, b=None,
                 use_bias=True):

        self.input = input
        self.activation = activation

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

        if use_bias:
            lin_output = tensor.dot(input, self.W) + self.b
        else:
            lin_output = tensor.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))

        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 dropout_rate, use_bias=True, activation=tensor.tanh,  W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)

def _dropout_from_layer(rng, layer, p):
    """p is the probability of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * tensor.cast(mask, theano.config.floatX)
    return output

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
        Since copying data into the GPU is slow, copying a minibatch every time
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
        # lets us get around this issue
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

class ConvNet(object):
    def __init__(self,
            rng,
            input,
            use_bias,
            kernel_position_product,
            batch_size,
            filter_size,
            pool_size,
            n_convpool_layers,
            n_hidden_layers,
            dropout,
            input_dropout,
            convpool_dropout,
            hidden_dropout,
            n_hidden_units,
            convpool_layer_activation,
            hidden_layer_activation,
            image_dim,
            n_output_dim,
            squared_filter_length_limit=15
            ):

        # determine the activation functions
        if convpool_layer_activation=='tanh':
            convpool_layer_activation=tensor.tanh
        else:
            raise NotImplementedError

        if hidden_layer_activation=='tanh':
            hidden_layer_activation=tensor.tanh
        elif hidden_layer_activation=='relu':
            hidden_layer_activation=relu
        else:
            raise NotImplementedError

        # Set up all the convolutional pooling layers
        input_size = (image_dim, image_dim)

        self.conv_pool_layers = []
        if dropout:
            self.dropout_conv_pool_layers = []
            # construct the dropout rate list
            dropout_rates = [input_dropout]
            dropout_rates.extend([convpool_dropout for i in range(n_convpool_layers)])
            dropout_rates.extend([hidden_dropout for i in range(n_hidden_layers)])

        if kernel_position_product < 10000:
            print 'Too few kernels in the input layer.'
            raise Exception

        # the product of the number of features and the number of pixel positions should be constant
        pixel_positions = (input_size[0] - filter_size[0] + 1)**2
        nkerns_current = int(kernel_position_product / pixel_positions)

        for layer_num in range(n_convpool_layers):

            if layer_num == 0: # first convpool layer
                next_layer_input = input.reshape((batch_size, 1, input_size[0], input_size[1]))
                image_shape=(batch_size, 1, input_size[0], input_size[1])
                filter_shape=(nkerns_current, 1, filter_size[0], filter_size[1])
                if dropout:
                    # dropout the input
                    next_dropout_layer_input = _dropout_from_layer(rng, next_layer_input, p=dropout_rates[0])

            else: # deeper convpool layer
                next_layer_input = self.conv_pool_layers[layer_num-1].output
                image_shape=(batch_size, nkerns_previous, input_size[0], input_size[1])
                filter_shape=(nkerns_current, nkerns_previous, filter_size[0], filter_size[1])
                if dropout:
                    next_dropout_layer_input = self.dropout_conv_pool_layers[layer_num-1].output

            if dropout:
               self.dropout_conv_pool_layers.append(DropoutConvPoolLayer(
                   rng=rng,
                   input=next_dropout_layer_input,
                   activation=convpool_layer_activation,
                   image_shape=image_shape,
                   filter_shape=filter_shape,
                   poolsize=pool_size,
                   use_bias=use_bias,
                   dropout_rate=dropout_rates[layer_num+1]
                   )
               )

            self.conv_pool_layers.append(ConvPoolLayer(
                rng=rng,
                input=next_layer_input,
                image_shape=image_shape,
                filter_shape=filter_shape,
                poolsize=pool_size,
                use_bias=use_bias,
                activation=convpool_layer_activation,
                W=self.dropout_conv_pool_layers[layer_num].W * (1 - dropout_rates[layer_num]) if dropout else None,
                b=self.dropout_conv_pool_layers[layer_num].b if dropout else None
                )
            )

            input_size = ((input_size[0] - filter_size[0] + 1) / pool_size[0],
                      (input_size[1] - filter_size[1] + 1) / pool_size[1])

            pixel_positions = (input_size[0] - filter_size[0] + 1)**2
            nkerns_previous = nkerns_current
            nkerns_current = int(kernel_position_product / pixel_positions)

        nkerns = nkerns_previous

        # Set up all the hidden layers
        #TODO: allow for different number of hidden units per layer
        hidden_layer_sizes = [n_hidden_units for i in range(n_hidden_layers)]
        hidden_layer_sizes.append(n_output_dim)
        hidden_layer_weight_matrix_sizes = zip(hidden_layer_sizes, hidden_layer_sizes[1:])

        self.hidden_layers = []
        self.dropout_hidden_layers = []

        next_layer_input = self.conv_pool_layers[-1].output.flatten(2)
        if dropout is True:
            next_dropout_layer_input = self.dropout_conv_pool_layers[-1].output.flatten(2)

        layer_counter = 0
        for n_in, n_out in hidden_layer_weight_matrix_sizes:
            if layer_counter == 0: # first hidden layer
                n_in = nkerns * input_size[0] * input_size[1]
            if dropout:
                next_dropout_layer = DropoutHiddenLayer(
                        rng=rng,
                        input=next_dropout_layer_input,
                        activation=hidden_layer_activation,
                        n_in=n_in, n_out=n_out, use_bias=use_bias,
                        dropout_rate=dropout_rates[layer_counter+n_convpool_layers+1])
                self.dropout_hidden_layers.append(next_dropout_layer)
                next_dropout_layer_input = next_dropout_layer.output

            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=hidden_layer_activation,
                    # scale the weight matrix W with (1-p)
                    W=next_dropout_layer.W * (1 - dropout_rates[layer_counter+n_convpool_layers]) if dropout else None,
                    b=next_dropout_layer.b if dropout else None,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.hidden_layers.append(next_layer)
            next_layer_input = next_layer.output

            layer_counter += 1

        # Set up the output layer
        if dropout:
            self.dropout_output_layer = LogisticRegression(
                    input=next_dropout_layer_input,
                    n_in=n_out, n_out=n_output_dim)

        self.output_layer = LogisticRegression(
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=self.dropout_output_layer.W * (1 - dropout_rates[-1]) if dropout else None,
            b=self.dropout_output_layer.b if dropout else None,
            n_in=n_out, n_out=n_output_dim)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        if dropout:
            self.dropout_negative_log_likelihood = self.dropout_output_layer.negative_log_likelihood
            self.dropout_errors = self.dropout_output_layer.errors

        self.negative_log_likelihood = self.output_layer.negative_log_likelihood
        self.errors = self.output_layer.errors
        self.y_pred = self.output_layer.y_pred

        # Grab all the layers and parameters together.
        if dropout:
            self.dropout_layers = list(itertools.chain(self.dropout_conv_pool_layers, self.dropout_hidden_layers, [self.dropout_output_layer]))
            self.params = [ param for layer in self.dropout_layers for param in layer.params ]
            assert len(self.dropout_layers) == n_convpool_layers + n_hidden_layers + 1
        else:
            self.layers = list(itertools.chain(self.conv_pool_layers, self.hidden_layers, [self.output_layer]))
            self.params = [ param for layer in self.layers for param in layer.params ]
            assert len(self.layers) == n_convpool_layers + n_hidden_layers + 1

def evaluate_convnet(
        kernel_position_product=20000,
        filter_size = (3, 3),
        pool_size = (2, 2),
        n_convpool_layers = 1,
        n_hidden_layers = 1,
        n_hidden_units = 100,
        convpool_layer_activation='tanh',
        hidden_layer_activation='relu',
        image_dim=32,
        n_output_dim=8,
        initial_learning_rate=0.8,
        learning_rate_decay=0.98,
        n_epochs=500,
        patience=5000,
        patience_increase=2,
        improvement_threshold=0.995,
        batch_size=100,
        training_data=None,
        validation_data=None,
        test_data=None,
        squared_filter_length_limit = 15.0,
        mom_params={"start": 0.5,
                    "end": 0.99,
                    "interval": 500},
        dropout=False,
        input_dropout=0.2,
        convpool_dropout=0.0,
        hidden_dropout=0.5,
        use_bias=True,
        random_seed=1234
    ):

    #print 'Kernel position product: ', kernel_position_product
    #print 'Filter size: ', filter_size
    #print 'Pool size: ', pool_size
    #print 'Number of convolutional pooling layers: ', n_convpool_layers
    #print 'Number of hidden layers: ', n_hidden_layers
    #print 'Number of hidden units: ', n_hidden_units
    #print 'ConvPool layer activation: ', convpool_layer_activation
    #print 'Hidden layer activation: ', hidden_layer_activation
    #print 'Initial learning rate: ', initial_learning_rate
    #print 'Learning rate decay: ', learning_rate_decay
    #print 'Batch size: ', batch_size
    #print 'Momentum parameters: ', mom_params
    #print 'Dropout enabled?: ', dropout
    #print 'Dropout rates: ', 'input', input_dropout, 'convpool dropout', convpool_dropout, 'hidden dropout', hidden_dropout

    # extract the params for momentum
    mom_start = mom_params["start"]
    mom_end = mom_params["end"]
    mom_epoch_interval = mom_params["interval"]

    if training_data is None or validation_data is None:
        print "No dataset given."
        sys.exit(1)

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

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = tensor.lscalar()    # index to a [mini]batch
    epoch = tensor.scalar()
    x = tensor.matrix('x')  # the data is presented as rasterized images
    y = tensor.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))

    rng = np.random.RandomState(random_seed)

    # construct the net
    conv_net = ConvNet(rng=rng,
                       input=x,
                       use_bias=use_bias,
                       kernel_position_product=kernel_position_product,
                       batch_size=batch_size,
                       filter_size=filter_size,
                       pool_size=pool_size,
                       n_convpool_layers=n_convpool_layers,
                       n_hidden_layers=n_hidden_layers,
                       dropout=dropout,
                       input_dropout=input_dropout,
                       convpool_dropout=convpool_dropout,
                       hidden_dropout=hidden_dropout,
                       n_hidden_units=n_hidden_units,
                       convpool_layer_activation=convpool_layer_activation,
                       hidden_layer_activation=hidden_layer_activation,
                       image_dim=image_dim,
                       n_output_dim=n_output_dim)

    # Build the expression for the cost function.
    cost = conv_net.negative_log_likelihood(y)
    if dropout:
        dropout_cost = conv_net.dropout_negative_log_likelihood(y)

    # compile functions to give predictions
    training_predictions = theano.function(
        inputs=[index],
        outputs=[conv_net.y_pred],
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
        }
    )

    validation_predictions = theano.function(
        inputs=[index],
        outputs=[conv_net.y_pred],
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
        }
    )

    if test_data is not None:
        test_predictions = theano.function(
            inputs=[index],
            outputs=[conv_net.y_pred],
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
            }
        )

    # create a function to compute the mistakes that are made by the model
    training_model = theano.function(
        [index],
        conv_net.errors(y),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        conv_net.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    #TODO: what is this?
    #theano.printing.pydotprint(test_model, outfile="test_file.png",
    #        var_with_name_simple=True)

    # Compute gradients of the model wrt parameters
    gparams = []
    for param in conv_net.params:
        # Use the right cost function here to train with or without dropout.
        gparam = tensor.grad(dropout_cost if dropout else cost, param)
        gparams.append(gparam)

    # ... and allocate memory for momentum'd versions of the gradient
    gparams_mom = []
    for param in conv_net.params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)

    # Compute momentum for the current epoch
    mom = ifelse(epoch < mom_epoch_interval,
            mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
            mom_end)

    # Update the step direction using momentum
    updates = OrderedDict()
    for gparam_mom, gparam in zip(gparams_mom, gparams):

        # change the update rule to match Hinton's dropout paper
        updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam

    # ... and take a step along that direction
    for param, gparam_mom in zip(conv_net.params, gparams_mom):
        # since we have included learning_rate in gparam_mom, we don't need it
        # here
        stepped_param = param + updates[gparam_mom]

        # This is a silly hack to constrain the norms of the rows of the weight
        # matrices.  This just checks if there are two dimensions to the
        # parameter and constrains it if so... maybe this is a bit silly but it
        # should work for now.
        if param.get_value(borrow=True).ndim == 2:
            #squared_norms = tensor.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
            #scale = tensor.clip(tensor.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
            #updates[param] = stepped_param * scale

            # max norm regularisation
            # constrain the norms of the COLUMNs of the weight, according to
            # https://github.com/BVLC/caffe/issues/109
            col_norms = tensor.sqrt(tensor.sum(tensor.sqr(stepped_param), axis=0))
            desired_norms = tensor.clip(col_norms, 0, tensor.sqrt(squared_filter_length_limit))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param


    # Compile theano function for training.  This returns the training cost and
    # updates the model parameters.
    output = dropout_cost if dropout else cost
    train_model = theano.function(
        inputs=[epoch, index],
        outputs=output,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # function for decaying the learning rate only after each epoch (not minibatch)
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

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    epoch_counter = 0
    done_looping = False
    start_time = time.clock()

    #results_file = open(results_file_name, 'wb')

    while (epoch_counter < n_epochs) and (not done_looping):
        try:
            # Train this epoch
            epoch_counter = epoch_counter + 1

            for minibatch_index in xrange(n_train_batches):
                iter = (epoch_counter - 1) * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print 'training @ iter = ', iter

                minibatch_avg_cost = train_model(epoch_counter, minibatch_index)

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
                          (epoch_counter, minibatch_index + 1, n_train_batches,
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

                        # test it on the test set
                        if test_data is not None:
                            test_pred = [
                                test_predictions(i) for i
                                in xrange(n_test_batches)
                            ]
                            test_pred = list(itertools.chain.from_iterable(test_pred))
                            print 'Wrote test predictions to file.\n'

                if patience <= iter:
                    done_looping = True
                    break

            decay_learning_rate()

        except KeyboardInterrupt:
            break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    if test_data is not None:
        return best_validation_loss, test_pred

    else:
        return best_validation_loss

if __name__ == '__main__':

    labeled_training, labeled_training_labels = util.load_labeled_training(flatten=True)
    assert labeled_training.shape == (2925, 1024)

    labeled_training -= np.mean(labeled_training) # global contrast normalisation

    test_images = util.load_public_test(flatten=True)
    test_images -= np.mean(labeled_training) # global contrast normalisation

    from zca import ZCA
    zca = ZCA().fit(labeled_training)
    labeled_training = zca.transform(labeled_training)
    #render_matrix(labeled_training[:100,:], flattened=True)

    #render_matrix(labeled_training[:100,:], flattened=True)
    #labeled_training = global_contrast_normalize(labeled_training, use_std=True)
    #render_matrix(labeled_training[:100,:], flattened=True)

    # dumb validation set partition for now
    util.shuffle_in_unison(labeled_training, labeled_training_labels)
    valid_split = labeled_training.shape[0] // 12
    train_data, train_labels = (labeled_training[valid_split:, :], labeled_training_labels[valid_split:])
    valid_data, valid_labels = (labeled_training[:valid_split, :], labeled_training_labels[:valid_split])

    _, test_labels = evaluate_convnet(
            training_data=(train_data, train_labels),
            validation_data=(valid_data, valid_labels),
            test_data=test_images
            )

    util.write_results(test_labels, 'predictions.csv')

