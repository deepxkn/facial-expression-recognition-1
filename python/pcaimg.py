from util import load_labeled_training, load_unlabeled_training
import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import cross_validation
plt.ion()


def pcaimg(X, k):
    """
    PCA matrix X to k dimensions.
    Inputs:
        X : Each column of X contains a data vector.
        k : Number of dimensions to reduce to.
    Returns:
        v : The eigenvectors. Each column of v is an eigenvector.
        mean : mean of X.
        projX : X projected down to k dimensions.
  """
    xdim, ndata = X.shape
    mean = np.mean(X, axis=1).reshape(-1, 1)
    X = X - mean
    cov = np.dot(X, X.T) / ndata

    w, v = lin.eigh(cov, eigvals=(xdim - k, xdim - 1))
    # w contains top k eigenvalues in increasing order of magnitude.
    # v contains the eigenvectors corresponding to the top k eigenvalues.

    plt.scatter([i for i in xrange(w.shape[0])], w)
    plt.draw()
    raw_input('Press Enter.')

    projX = np.dot(v.T, X)
    return v, mean, projX


def ShowEigenVectors(v):
    """Displays the eigenvectors as images in decreasing
    order of eigen value."""
    plt.figure(1)
    plt.clf()
    for i in xrange(v.shape[1]):
        plt.subplot(1, v.shape[1], i+1)
        plt.imshow(v[:, v.shape[1] - i - 1].reshape(16, 16).T,
                   cmap=plt.cm.gray)
    plt.draw()
    raw_input('Press Enter.')


def main():
    K = 30  # Number of dimensions to PCA down to.
    test_image, test_label = load_labeled_training(flatten=True)
    train_image = load_unlabeled_training(flatten=True)
    pca = PCA(n_components=K).fit(test_image)
    proj_test = pca.transform(test_image)
    #v, mean, projX = pcaimg(train_image.T, K)
    #proj_test = np.dot(v.T, test_image.T).T
    #assert proj_test.shape[0] == test_label.shape[0]
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        proj_test, test_label)
    nbrs = KNeighborsClassifier(n_neighbors=1)
    scores = cross_validation.cross_val_score(nbrs, proj_test, test_label, cv=3)
    print scores
    # nbrs = KNeighborsClassifier(n_neighbors=1).fit(x_train, y_train)
    # knn_labels = nbrs.predict(x_test)
    # print float(sum(l == k for k, l in zip(knn_labels, y_test)))/len(knn_labels)

if __name__ == '__main__':
    main()
