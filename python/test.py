import sys
import os
import util
import scipy.io
from sklearn import datasets

def main():
    images = util.load_unlabeled_training()
    print images.shape 

if __name__ == '__main__':
    main()
