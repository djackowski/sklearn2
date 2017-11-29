from sklearn.metrics import accuracy_score
import math


def scorer_01loss(estimator, X, y):
    y_pred = estimator.predict(X)
    print("Accuracy = ", accuracy_score(y, y_pred))
    return 1 - accuracy_score(y, y_pred)


def scorer_squared_error(estimator, X, y):
    probabs = estimator.predict_proba(X)

    index_of_0 = 0
    if estimator.classes_[index_of_0] == 1:
        index_of_0 = 1

    inst = X.shape[0]
    if inst == 0:
        return 0.0

    s = 0.0
    for i in range(inst):
        var = int((y[i] + index_of_0) % 2)
        try:
            p = probabs[i, var]
        except IndexError:
            continue
        s += math.pow(1.0 - p, 2)
    return s / inst

