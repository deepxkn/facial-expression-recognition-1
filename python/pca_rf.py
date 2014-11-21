import os
import sys
from util import load_pca_proj, shuffle_in_unison, load_pca_test, write_results
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

def main():
    N_TREE = 101
    k = 100
    rfc = RandomForestClassifier(n_estimators=N_TREE, max_depth=2)
    proj_test, labels = load_pca_proj(K=k)
    shuffle_in_unison(proj_test, labels)
    scores = cross_validation.cross_val_score(rfc, proj_test, labels, cv=10)
    pt = load_pca_test(K=k)
    rfc.fit(proj_test, labels)
    pred = rfc.predict(pt)
    write_results(pred, '../rfc_res.csv')
    print scores
    print np.mean(scores)
    print np.var(scores)

if __name__ == '__main__':
    sys.exit(main())
