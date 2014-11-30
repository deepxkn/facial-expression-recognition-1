import os
import sys
from util import load_pca_proj, shuffle_in_unison, load_pca_test, write_results
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
def main():
    pk=80
    for pk in [80, 200, 500, 1024]:
        proj_test, labels = load_pca_proj(K=pk)
        shuffle_in_unison(proj_test, labels)
        for k in [1,2,3,4,5,6,7,8,9,15,20,25,30]:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_validation.cross_val_score(knn, proj_test, labels, cv=10)
            print "K: " + str(k)
            print "PK: " + str(pk)
            print scores
            print np.mean(scores)
            print np.var(scores)

if __name__ == '__main__':
    sys.exit(main())
