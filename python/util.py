import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
        images = images.reshape(n, x*y)
        #images = images.reshape(-1, images.shape[0]).T
        assert images.shape == (n, x*y)

    return images, labels

def load_unlabeled_training(flatten=False):
    unlabeled = scipy.io.loadmat('../unlabeled_images.mat')
    images = unlabeled['unlabeled_images']

    # permute dimensions so that the number of instances is first
    x, y, n = images.shape
    images = np.transpose(images, [2, 0, 1])
    assert images.shape == (n, x, y)

    # flatten the pixel dimensions
    if flatten is True:
        n, x, y = images.shape
        images = images.reshape(-1, images.shape[0]).T
        assert images.shape == (n, x*y)

    return images

"""The following function is adapted from A3.
"""

def render_matrix(matrix, ncols=15, show=True):
    """Show matrix as a set of images.
    Plot images in row-major order.
    """
    if matrix.shape[0] > 200:
        print('Too many images to render efficiently.')
        return

    nrows = matrix.shape[0]//ncols+1

    plt.clf()
    plt.figure(1)
    fig, axs = plt.subplots(nrows, ncols)
    axs = axs.ravel()

    for i in xrange(matrix.shape[0]):
        axs[i].imshow(matrix[i, :, :], cmap = cm.Greys_r,
            interpolation='none') # for no anti-aliasing
        axs[i].axis('off')

    # clear empty subplots
    for i in xrange(matrix.shape[0], nrows*ncols):
        fig.delaxes(axs[i])

    if show:
        plt.show()

"""The following function is from
http://deeplearning.net/tutorial/code/utils.py.
"""

def scale_to_unit_interval(ndar, eps=1e-20):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

if __name__ == '__main__':
    data = load_labeled_training()
    render_matrix(data[0][:100, : , :])
