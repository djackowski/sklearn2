import time
import timeit
from collections import defaultdict

import numpy as np
from sklearn.utils import shuffle

from utils.evaluate import scorer_squared_error, scorer_01loss


def evaluate_accuracy_and_time(classifier, X_train, y_train, X_test, y_test, start, stop):
    start_time = timeit.default_timer()
    # X_shuf, Y_shuf = shuffle(X_transformed, Y)
    print("Unique: ", np.unique(y_train))
    classifier.fit(X_train, y_train)

    training_time = timeit.default_timer() - start_time
    print("Training time = {0}".format(training_time))
    # scorers = [(scorer_01loss, "0/1 loss"), (scorer_squared_error, "squared error")]
    scorersLoss = [(scorer_01loss, "0/1 loss")]
    scorersError = [(scorer_squared_error, "squared error")]

    lossResult = 0
    errorResult = 0
    lossTest = 0
    errorTest = 0

    # for scorer, scorer_name in scorers:
    #     result = scorer(classifier, X_train[start:stop], y_train[start:stop])
    #     print("Train {0} = {1}".format(scorer_name, result))
    #
    # for scorer, scorer_name in scorers:
    #     result = scorer(classifier, X_test, y_test)
    #     print("Test {0} = {1}".format(scorer_name, result))
    #
    for scorer, scorer_name in scorersLoss:
        resultTrain = scorer(classifier, X_train, y_train)
        print("Train {0} = {1}".format(scorer_name, resultTrain))
        resultTest = scorer(classifier, X_test, y_test)
        print("Test {0} = {1}".format(scorer_name, resultTest))
        lossResult = resultTrain
        lossTest = resultTest

    for scorer, scorer_name in scorersError:
        resultTrain = scorer(classifier, X_train, y_train)
        print("Train {0} = {1}".format(scorer_name, resultTrain))
        print(X_train.shape[0])
        resultTest = scorer(classifier, X_test, y_test)
        print("Test {0} = {1}".format(scorer_name, resultTest))
        errorResult = resultTrain
        errorTest = resultTest

    return errorResult, lossResult, training_time, lossTest, errorTest
