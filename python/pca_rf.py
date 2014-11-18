import os
import sys
from util import load_pca_proj, shuffle_in_unison
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

def main():
    N_TREE = 101
    rfc = RandomForestClassifier(n_estimators=N_TREE)
    proj_test, labels = load_pca_proj()
    shuffle_in_unison(proj_test, labels)
    scores = cross_validation.cross_val_score(rfc, proj_test, labels, cv=10)
    print scores

if __name__ == '__main__':
    sys.exit(main())
