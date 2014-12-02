import os
import sys
from util import load_pca_proj, shuffle_in_unison, load_pca_test, write_results, load_labeled_training, standardize, load_pca_hidden
import numpy as np
from sklearn.svm import SVC
from sklearn import cross_validation

def main():
    k=500
      #images, labels = load_labeled_training(flatten=True)
   #images = standardize(images)
   #shuffle_in_unison(images, labels)
   #for k in [100,500,1024]:
      #proj_test, labels = load_pca_proj(K=k)
      #shuffle_in_unison(proj_test, labels)
      #for ker in ['linear', 'sigmoid', 'rbf']:
           #svc = SVC(kernel=ker)
           #scores = cross_validation.cross_val_score(svc, proj_test, labels, cv=10)
           #print "Kernel: " + ker
           #print "K: " + str(k)
           #print scores
           #print np.mean(scores)
           #print np.var(scores)
    proj_test, labels = load_pca_proj(K=k)
    shuffle_in_unison(proj_test, labels)
    svc = SVC()
    scores = cross_validation.cross_val_score(svc, proj_test, labels, cv=10)
    pt = load_pca_hidden(K=k)
    svc.fit(proj_test, labels)
    pred = svc.predict(pt)
    write_results(pred, '../svm_res.csv')
   #pt = load_pca_test(K=k)
   #svc.fit(proj_test, labels)
   #pred = svc.predict(pt)
   #write_results(pred, '../svm_res.csv')

if __name__ == '__main__':
    sys.exit(main())
