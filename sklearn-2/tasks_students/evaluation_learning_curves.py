'''
Author: Kalina Jasinska
'''
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from classifiers_students.naive_bayes import NaiveBayesGaussian
from tasks_students.plot_learning_curve import evaluate_accuracy_and_time
from utils.load import read_datasets

# Implement plotting of a learning curve using sklearn
# Remember that the evaluation metrics to plot are 0/1 loss and squared error

# ('data/badges2-train.csv', 'data/badges2-test.csv', "Badges2")
datasets = [
    ('data/credit-a-train.csv', 'data/credit-a-test.csv', "Credit-a"),
    ('data/credit-a-mod-train.csv', 'data/credit-a-mod-test.csv', "Credit-a-mod"),
    ('data/spambase-train.csv', 'data/spambase-test.csv', "Spambase"),
    ('data/grypa-train.csv', 'data/grypa-test.csv', "Grypa")
]

classifiers = [(GaussianNB(), 'GaussianNB'),
               (LogisticRegression(), 'LogisticRegression'),
               (NaiveBayesGaussian(), 'NaiveBayesGaussian')]


def make_learning_curves():
    raise NotImplementedError


def drawRange(X_train, y_train):
    drawn = np.sort(np.random.choice(len(X_train), 2))
    start, stop = drawn
    if start == stop:
        return drawRange(X_train, y_train)
    elif stop - start < 4:
        return drawRange(X_train, y_train)
    elif checkIfYtrainHasOnlyOneClass(y_train):
        return drawRange(X_train, y_train)
    else:
        return drawn


def checkIfYtrainHasOnlyOneClass(y_train):
    if len(np.unique(y_train)) < 2:
        return True
    else:
        return False

def evaluate_classifier():
    for data in datasets:
        fn, fn_test, ds_name = data
        print("\n")
        print("Data: ", ds_name)
        print("\n")
        for classifierData in classifiers:
            X_train, y_train, X_test, y_test, is_categorical = read_datasets(fn, fn_test)
            classifier, classifierName = classifierData
            print("\n")
            print("-----NEW CLASSIFIER INCOMING-----")
            print("Classifier: ", classifierName)
            scorerTrainResultLoss = dict()
            scorerTrainResultError = dict()
            times = dict()
            for i in range(0, 500):
                start, stop = drawRange(X_train, y_train)
                examplesNo = stop
                print("\n")
                print("Examples Number: ", examplesNo)
                print("\n")
                print("Start: ", start, "Stop: ", stop)
                scorerTrainResultError[examplesNo], scorerTrainResultLoss[examplesNo], times[examplesNo] = evaluate_accuracy_and_time(
                    classifier, X_train[0:stop], y_train[0:stop], X_test, y_test, start, stop)
            listsLoss = sorted(scorerTrainResultLoss.items())
            listsErrors = sorted(scorerTrainResultError.items())
            listsTimes = sorted(times.items())
            xL, yL = zip(*listsLoss)
            xE, yE = zip(*listsErrors)
            xT, yT = zip(*listsTimes)

            plt.figure(1).set_size_inches(10, 10)
            plt.subplot(211)
            plt.title(classifierName + " " + ds_name)
            plt.grid(True)
            plt.plot(xL, yL, label="Loss")
            plt.plot(xE, yE, label="Error")
            plt.ylabel("Loss/Error")
            plt.xlabel("Training Data")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=2, mode="expand", borderaxespad=0.)
            plt.subplot(212)
            plt.plot(xT, yT)
            plt.ylabel("Loss/Error")
            plt.xlabel("Time")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=2, mode="expand", borderaxespad=0.)
            plt.show()


if __name__ == "__main__":
    evaluate_classifier()
    # make_learning_curves()
