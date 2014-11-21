import os
import sys
from util import standardize, shuffle_in_unison, load_labeled_training
from pybrain.tools import shortcuts
from pybrain.tools import validation
from pybrain.structure import SoftmaxLayer
from pybrain.utilities import percentError
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import Validator
def main():
    images, labels = load_labeled_training(flatten=True)
    images = standardize(images)
    shuffle_in_unison(images, labels)
    ds = ClassificationDataSet(images.shape[1],1, nb_classes=7)
    for i,l in zip(images, labels):
        ds.addSample(i,[l-1])
    #ds._convertToOneOfMany()
    test, train = ds.splitWithProportion(0.2)
    test._convertToOneOfMany()
    train._convertToOneOfMany()
    net=shortcuts.buildNetwork(1024, 100, train.outdim, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(net, dataset=train, momentum=0.1, learningrate=0.1)
    #cv = validation.CrossValidator(trainer, ds)
    #print cv.validate()
    net.randomize()
    tr_labels_2 = net.activateOnDataset(train).argmax(axis=1)
    trnres = percentError(tr_labels_2, train['class'])
    #trnres = percentError(trainer.testOnClassData(dataset=train), train['class'])
    testres = percentError(trainer.testOnClassData(dataset=test), test['class'])
    print "Training error: %.10f, Test error: %.10f" % (trnres, testres)
    print "Iters: %d" % trainer.totalepochs

    for i in range(20):
        trainer.trainEpochs(10)
        trnres = percentError(trainer.testOnClassData(dataset=train), train['class'])
        testres = percentError(trainer.testOnClassData(dataset=test), test['class'])
        print "Iteration: %d, Training error: %.10f, Test error: %.10f" % (trainer.totalepochs, trnres, testres)
    

if __name__ == '__main__':
    sys.exit(main())
