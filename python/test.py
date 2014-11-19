import sys
import os
import util
from zca import ZCA
from collections import defaultdict
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

def main():
    util.load_public_test()
    images,labels = util.load_labeled_training(flatten=True)
    #images,labels = util.load_pca_proj(K=100)
    #zca = ZCA().fit(tr_images)
    #std_x = zca.transform(images)
    images -= np.mean(images)
    #images /= np.sqrt(np.var(images))
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
