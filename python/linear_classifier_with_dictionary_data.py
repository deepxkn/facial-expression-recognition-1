from time import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

from sklearn.svm import SVC
from sklearn import cross_validation
from scipy.stats import mode

import util
import zca

def get_dictionary_data(n_comp=20, zero_index=True):
    unlabeled = util.load_unlabeled_training(flatten=False)
    height, width = 32, 32
    n_images = 10000
    patch_size = (8, 8)

    unlabeled = util.standardize(unlabeled)
    np.random.shuffle(unlabeled)

    print('Extracting reference patches...')

    patches = np.empty((0, 64))
    t0 = time()

    for image in unlabeled[:n_images, :, :]:
        data = np.array(extract_patches_2d(image, patch_size, max_patches=0.10))
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

    #plt.figure(figsize=(4.2, 4))
    #for i, comp in enumerate(V[:100]):
    #    plt.subplot(10, 10, i + 1)
    #    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
    #               interpolation='nearest')
    #    plt.xticks(())
    #    plt.yticks(())
    #plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    #plt.show()

    labeled_data, labels = util.load_labeled_training(flatten=False, zero_index=True)
    labeled_data = util.standardize(labeled_data)

    test_data = util.load_all_test(flatten=False)
    test_data = util.standardize(test_data)

    #util.render_matrix(test_data, flattened=False)

    print('Training SVM with the training images...')
    t0 = time()
    reconstructed_images = np.empty((0, 64))
    multiplied_labels = np.empty((0))

    for i in range(len(labeled_data)):
        image = labeled_data[i, :, :]
        label = labels[i]
        data = extract_patches_2d(image, patch_size, max_patches=0.50)
        data = data.reshape(data.shape[0], -1)
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0) + 1e-20

        code = dico.transform(data)
        patches = np.dot(code, V)
        z.transform(patches)

        reconstructed_images = np.concatenate([reconstructed_images, patches])
        extended_labels = np.asarray([label] * len(patches))
        multiplied_labels = np.concatenate([multiplied_labels, extended_labels])

    print(reconstructed_images.shape, multiplied_labels.shape)
    svc = SVC()
    #print('Getting cross-val scores...')
    #scores = cross_validation.cross_val_score(svc, reconstructed_images, multiplied_labels, cv=10)
    #print('cross-val scores:', scores)
    #print('cross-val mean:', np.mean(scores))
    #print('cross-val variance:', np.var(scores))

    print('done in %.2fs.' % (time() - t0))

    svc.fit(reconstructed_images, multiplied_labels)

    print('Reconstructing the test images...')
    t0 = time()

    predictions = []

    for i, image in enumerate(test_data):
        data = extract_patches_2d(image, patch_size, max_patches=0.25)
        data = data.reshape(data.shape[0], -1)
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0) + 1e-20

        code = dico.transform(data)
        patches = np.dot(code, V)
        z.transform(patches)

        pred = svc.predict(patches)
        print('Variance in the predictions:', np.var(pred))
        predictions.append(mode(pred))

    print('done in %.2fs.' % (time() - t0))

    predictions += 1
    util.write_results(predictions, 'svm_patches_25_percent_20_comp.csv')

if __name__ == '__main__':
    get_dictionary_data(n_comp=20)
