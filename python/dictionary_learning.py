from time import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import DictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

import util
import zca

def get_dictionary_data(n_comp=20):
    unlabeled = util.load_unlabeled_training(flatten=False)
    height, width = 32, 32
    n_images = 5000
    patch_size = (8, 8)

    np.random.shuffle(unlabeled)

    print('Extracting reference patches...')

    patches = np.empty((0, 64))

    for image in unlabeled[:n_images, :, :]:
        t0 = time()
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

    labeled_data, labels = util.load_labeled_training(flatten=False)

    print('Reconstructing the images...')
    t0 = time()
    reconstructed_images = np.empty((0, 32, 32))

    for image in labeled_data:
        data = extract_patches_2d(image, patch_size)
        data = data.reshape(data.shape[0], -1)
        data -= np.mean(data, axis=0)

        code = dico.transform(data)
        patches = np.dot(code, V)
        print patches.shape
        z.transform(patches)
        patches = patches.reshape(len(data), *patch_size)

        data = reconstruct_from_patches_2d(patches, (width, height))
        data = data.reshape(1, 32, 32)
        reconstructed_images = np.concatenate([reconstructed_images, data])

    print('done in %.2fs.' % (time() - t0))

    # flatten
    n, x, y = reconstructed_images.shape
    images = reconstructed_images.reshape(reconstructed_images.shape[0], reconstructed_images.shape[1]*reconstructed_images.shape[2])
    assert reconstructed_images.shape == (n, x*y)

    util.render_matrix(reconstructed_images[:20, :, :][:20, :, :][:20, :, :][:20, :, :][:20, :, :][:20, :, :][:20, :, :][:20, :, :][:20, :, :][:20, :, :], flattened=False)

    return (reconstructed_images, labels)

if __name__ == '__main__':
    get_dictionary_data()
