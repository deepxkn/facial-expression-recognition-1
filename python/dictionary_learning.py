from time import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

import util
import zca

def get_dictionary_data(n_comp=20, zero_index=False):
    unlabeled = util.load_unlabeled_training(flatten=False)
    height, width = 32, 32
    n_images = 5000
    patch_size = (8, 8)

    unlabeled = util.standardize(unlabeled)
    np.random.shuffle(unlabeled)

    print('Extracting reference patches...')

    patches = np.empty((0, 64))
    t0 = time()

    for image in unlabeled[:n_images, :, :]:
        data = np.array(extract_patches_2d(image, patch_size, max_patches=0.05))
        data = data.reshape(data.shape[0], -1)
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0) + 1e-20
        patches = np.concatenate([patches, data])

    print('done in %.2fs.' % (time() - t0))

    # whiten the patches
    z = zca.ZCA()
    z.fit(patches)
    z.transform(patches)

    print('Learning the dictionary...')
    t0 = time()
    dico = MiniBatchDictionaryLearning(n_components=n_comp, alpha=1)
    V = dico.fit(patches).components_
    dt = time() - t0
    print('done in %.2fs.' % dt)

    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(V[:100]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()

    labeled_data, labels = util.load_labeled_training(flatten=False, zero_index=True)
    labeled_data = util.standardize(labeled_data)

    test_data = util.load_hidden_test(flatten=False)
    test_data = util.standardize(test_data)

    #util.render_matrix(test_data, flattened=False)

#TODO
    labeled_data = labeled_data[:10, :, :]

    print('Reconstructing the training images...')
    t0 = time()
    reconstructed_images = np.empty((0, 32, 32))

    for image in labeled_data:
        data = extract_patches_2d(image, patch_size)
        data = data.reshape(data.shape[0], -1)
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0) + 1e-20

        code = dico.transform(data)
        patches = np.dot(code, V)
        z.transform(patches)
        patches = patches.reshape(len(data), *patch_size)

        data = reconstruct_from_patches_2d(patches, (width, height))
        data = data.reshape(1, 32, 32)
        reconstructed_images = np.concatenate([reconstructed_images, data])

    print('done in %.2fs.' % (time() - t0))

    # flatten
    n, x, y = reconstructed_images.shape
    training_images = reconstructed_images.reshape(reconstructed_images.shape[0], reconstructed_images.shape[1]*reconstructed_images.shape[2])
    assert training_images.shape == (n, x*y)

    #util.render_matrix(training_images, flattened=True)
#TODO
    test_data = test_data[:10, :, :]

    print('Reconstructing the test images...')
    t0 = time()
    reconstructed_test_images = np.empty((0, 32, 32))

    for image in test_data:
        data = extract_patches_2d(image, patch_size)
        data = data.reshape(data.shape[0], -1)
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0) + 1e-20

        code = dico.transform(data)
        patches = np.dot(code, V)
        z.transform(patches)
        patches = patches.reshape(len(data), *patch_size)

        data = reconstruct_from_patches_2d(patches, (width, height))
        data = data.reshape(1, 32, 32)
        reconstructed_test_images = np.concatenate([reconstructed_test_images, data])

    print('done in %.2fs.' % (time() - t0))

    # flatten
    n, x, y = reconstructed_test_images.shape
    test_images = reconstructed_test_images.reshape(reconstructed_test_images.shape[0], reconstructed_test_images.shape[1]*reconstructed_test_images.shape[2])
    assert test_images.shape == (n, x*y)

    #util.render_matrix(test_images, flattened=True)

    return (training_images, labels, test_images)

if __name__ == '__main__':
    get_dictionary_data()
