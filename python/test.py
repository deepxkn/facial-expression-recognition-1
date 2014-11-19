import sys
import os
import util
from zca import ZCA
from collections import defaultdict
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

def main():
    util.load_public_test()
    images,labels = util.load_labeled_training(flatten=True)
    print labels[0:20]
    print images
    images = images.astype(float)    
    #images = preprocessing.scale(images)
    print images[0][0]
    #images,labels = util.load_pca_proj(K=100)
    #zca = ZCA().fit(tr_images)
    #std_x = zca.transform(images)
    #for i in images:
        #for j in range(len(i)):
            #i[j] = float(i[j])
    mean = np.mean(images,axis=1)
    print mean.shape
    print mean[:20]
    sd = np.sqrt(np.var(images, axis=1) + 0.01)
    print sd
    #print sd.shape
    print images[0][0]
    for i in range(images.shape[0]):
        for j in range(len(images[i])):
            images[i][j] -= mean[i]
    print mean[0]
    print images[0][0]
    for i in range(images.shape[0]):
        for j in range(len(images[i])):
            images[i][j] /= sd[i]
    print images[0][0]
    util.shuffle_in_unison(images, labels)
    for k in [3,4,5,6,7,8,9,10, 15, 20, 25, 30, 40, 50]:
        print str(k) + ": " + '\n'
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, images, labels, cv=10)
        print score
        print np.mean(score)
    #d = defaultdict(lambda: 0)
    #for l in labels:
        #d[l] += 1
    #print d
    #print images.shape

if __name__ == '__main__':
    main()
