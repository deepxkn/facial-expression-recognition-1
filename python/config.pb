# Configuration file for spearmint hyperparameter optimisation of convnet
language: PYTHON
name: "optimise_convnet"

variable {
    name: "number of epochs"
    type: INT
    size: 1
    min: 100
    max: 100
}

variable {
    name: "batch size"
    type: INT
    size: 1
    min: 30
    max: 200
}


variable {
    name: "filter size"
    type: INT
    size: 1
    min: 3
    max: 5
}


variable {
    name: "number of kernels"
    type: ENUM
    size: 1
    options: "20,30,40"
    options: "30,50,100"
}

variable {
    name: "pool size"
    type: INT
    size: 1
    min: 2
    max: 2
}

variable {
    name: "learning rate"
    type: FLOAT
    size: 1
    min: 0.001
    max: 0.1
}

variable {
    name: "learning rate decay"
    type: FLOAT
    size: 1
    min: 0.98
    max: 0.999
}

variable {
    name: "convpool layer activation function"
    type: ENUM
    size: 1
    options: "tanh"
}

variable {
    name: "hidden layer activation function"
    type: ENUM
    size: 1
    options: "tanh"
    options: "relu"
}

variable {
    name: "number of convpool layers"
    type: INT
    size: 1
    min: 1
    max: 3
}

variable {
    name: "number of hidden layers"
    type: INT
    size: 1
    min: 1
    max: 3
}

variable {
    name: "number of hidden units"
    type: INT
    size: 1
    min: 30
    max: 500
}
