import convnet

parameters = {
    # hyperparameters
    'number of epochs' :            (10^x for x in range(1, 4)),
    'number of kernels' :           ([20, 50],),
    'batch size' :                  (5*10^x for x in range(1, 4)),
    'filter size' :                 ([5, 5],),
    'pool size' :                   ([2, 2],),
    'learning rate' :               (10^x for x in range(-4, 1)),
    'learning rate decay' :         (None, 0.999, 0.995, 0.98),
    'patience' :                    (10000,)
    'patiene increase' :            (2,),
    'improvement threshold' :       (0.995,)
    # architecture
    'number of convpool layers' :   (1, 2, 3),
    'number of hidden layers' :     (1, 2, 3),
    'activation function' :         ('tanh', 'ReLU'),

    }



if __name__ == '__main__':

