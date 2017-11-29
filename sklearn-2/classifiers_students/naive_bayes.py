import numpy as np
import math

from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from collections import defaultdict


class NaiveBayesNominal:
    def __init__(self):
        self.classes_ = [0, 1, 2, 3]
        self.model = defaultdict(lambda: defaultdict(dict))
        self.yesGrypaProb = 0
        self.noGrypaProb = 0
        self.probabies = []

    def fit(self, X, y):
        index = 0
        counter = 0
        yesGrypa = np.count_nonzero(y)
        noGrypa = len(y) - yesGrypa
        self.yesGrypaProb = yesGrypa / float(len(y))
        self.noGrypaProb = noGrypa / float(len(y))

        for k in range(0, len(y)):
            for i in range(0, len(X[0])):
                for j in range(0, len(y)):
                    tempX = X[k][index]
                    tempY = y[k]
                    if (tempX == X[j][index]) and (tempY == y[j]):
                        counter = counter + 1
                if tempY == 0:
                    counter = counter / float(noGrypa)
                else:
                    counter = counter / float(yesGrypa)
                self.model[self.classes_[i]][y[k]][X[k][index]] = counter
                counter = 0
                index = index + 1
            index = 0

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        eventual = []
        for i in range(0, len(X)):
            num = self.yesGrypaProb
            denom = self.noGrypaProb
            for j in range(0, len(X[i])):
                num *= self.model[j][1][X[i][j]]
                denom *= self.model[j][0][X[i][j]]
            result = num / (float(denom + num))
            if result > 0.5:
                eventual.append(1)
            else:
                eventual.append(0)
        return eventual


class NaiveBayesGaussian:
    def __init__(self):
        self.classes_ = []
        self.classIndexes = dict()
        self.predict_prob = []
        self.classesLength = defaultdict(dict)
        self.allY = 0
        self.allProbs = defaultdict(dict)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        epsilon = 1e-9 * np.var(X, axis=0).max()

        features = X.shape[1]
        classes = len(self.classes_)
        self.deviation = np.zeros((classes, features))
        self.mean = np.zeros((classes, features))

        for classElement in self.classes_:
            self.classIndexes[classElement] = np.where(y == classElement)[0]

        self.allY = sum([len(x) for x in self.classIndexes.values()])

        indexI = 0
        for i in self.classes_:
            Xclasses = []
            for j in self.classIndexes[i]:
                Xclasses.append(X[j])
            self.deviation[indexI, :] = (np.var(Xclasses, axis=0))
            self.mean[indexI, :] = (np.mean(Xclasses, axis=0))
            indexI += 1
        self.deviation[:, :] += epsilon

    def likelihood(self, X):
        lh = []
        for i in range(np.size(self.classes_)):
            pY = np.log(float((self.classesLength[i]) / float(self.allY)))
            deviation = self.deviation[i, :]
            sumLogPiSigma = np.sum(np.log(2. * np.pi * deviation))
            tempProbs = -0.5 * sumLogPiSigma
            tempProbs -= 0.5 * np.sum((np.asarray(X - self.mean[i, :]) ** 2) / (self.deviation[i, :]), 1)
            lh.append(pY + tempProbs)

        lh = np.array(lh).T

        return lh

    def predict(self, X):
        for i in self.classes_:
            self.classesLength[i] = len(self.classIndexes[i])

        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def predict_proba(self, X):
        log_prob_x = logsumexp(self.likelihood(X), axis=1)
        return np.exp(self.likelihood(X) - np.atleast_2d(log_prob_x).T)


class NaiveBayesNumNom(BaseEstimator):
    def __init__(self, is_cat=None, m=0.0):
        raise NotImplementedError

    def fit(self, X, yy):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
