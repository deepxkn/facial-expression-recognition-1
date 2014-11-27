import os
import sys
from util import standardize, shuffle_in_unison, load_labeled_training, load_pca_proj, load_public_test, write_results
from pybrain.tools import shortcuts
from pybrain.tools import validation
from pybrain.structure import SoftmaxLayer
from pybrain.utilities import percentError
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.tools.validation import Validator
def main():
    images, labels = load_labeled_training(flatten=True)
    public_test = load_public_test(flatten=True)
    images = standardize(images)
    #images, labels = load_pca_proj(K=100)
    shuffle_in_unison(images, labels)
    ds = ClassificationDataSet(images.shape[1],1, nb_classes=7)
    testset = ClassificationDataSet(public_test.shape[1])
    public_test=standardize(public_test)
    for i in public_test:
        testset.addSample(i,[0])
    for i,l in zip(images, labels):
        ds.addSample(i,[l-1])
    #ds._convertToOneOfMany()
    test, train = ds.splitWithProportion(0.2)
    test._convertToOneOfMany()
    train._convertToOneOfMany()
    net=shortcuts.buildNetwork(train.indim, 500, 1000,train.outdim, outclass=SoftmaxLayer)

    trainer = BackpropTrainer(net, dataset=train, learningrate=0.005, weightdecay=0.01)
    #trainer = RPropMinusTrainer(net, dataset=train)
    #cv = validation.CrossValidator(trainer, ds)
    #print cv.validate()
    net.randomize()
    tr_labels_2 = net.activateOnDataset(train).argmax(axis=1)
    trnres = percentError(tr_labels_2, train['class'])
    #trnres = percentError(trainer.testOnClassData(dataset=train), train['class'])
    testres = percentError(trainer.testOnClassData(dataset=test), test['class'])
    print "Training error: %.10f, Test error: %.10f" % (trnres, testres)
    print "Iters: %d" % trainer.totalepochs
    for i in range(10):
        trainer.trainEpochs(10)
        trnres = percentError(trainer.testOnClassData(dataset=train), train['class'])
        testres = percentError(trainer.testOnClassData(dataset=test), test['class'])
        trnmse = trainer.testOnData(dataset=train)
        testmse = trainer.testOnData(dataset=test)
        print "Iteration: %d, Training error: %.5f, Test error: %.5f" % (trainer.totalepochs, trnres, testres)
        print "Training MSE: %.5f, Test MSE: %.5f" % (trnmse, testmse)
    out=trainer.testOnClassData(dataset=testset)
    for i in range(len(out)):
        out[i] += 1
    write_results(out, 'nn_predictions.csv')
if __name__ == '__main__':
    sys.exit(main())
