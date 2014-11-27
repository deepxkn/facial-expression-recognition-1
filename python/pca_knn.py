import os
import sys
from util import load_pca_proj, shuffle_in_unison, load_pca_test, write_results
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
def main():
    k=200
    proj_test, labels = load_pca_proj(K=k)
    shuffle_in_unison(proj_test, labels)
    knn = KNeighborsClassifier()
    scores = cross_validation.cross_val_score(knn, proj_test, labels, cv=10)
    print scores

if __name__ == '__main__':
    sys.exit(main())
