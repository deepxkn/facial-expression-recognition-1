import scipy.io

def load_labeled_training():
    labeled = scipy.io.loadmat('../labeled_images.mat')
    labels = labeled['tr_labels']
    labels = [l[0] for l in labels]
    images = labeled['tr_images']
    return images, labels

def load_unlabeled_training():
    unlabeled = scipy.io.loadmat('../unlabeled_images.mat')
    images = unlabeled['unlabeled_images']
    return images
