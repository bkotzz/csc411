import numpy as np
from plot_digits import *
from utils import *
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import LabelKFold

def logistic_test(images, labels, test_images):
    logistic = linear_model.LogisticRegression()

    test_labels = test(images, labels, test_images, logistic)
    create_submission(test_labels)

def test(images, labels, test_images, model):
    return model.fit(images, labels).predict(test_images)

def svm_linear_validation(images, labels, ids):
    lkf = LabelKFold(ids, n_folds=2)

    scores = []
    for train, test in lkf:
        X = images[train]
        y = labels[train]

        X_test = images[test]
        y_test = labels[test]

        model_scores = []
        for C in [0.1, 0.5, 1.0, 1.5, 2.0]:
            model = svm.SVC(kernel='linear', C=C).fit(X, y) # high 60s for all C
            model_scores.append(validation_score(model, X_test, y_test))
        scores.append(model_scores)

    return scores

def svm_validation(images, labels, ids):

    lkf = LabelKFold(ids, n_folds=2)

    C = 1.0  # SVM regularization parameter

    scores = []
    for train, test in lkf:
        X = images[train]
        y = labels[train]

        X_test = images[test]
        y_test = labels[test]

        svc = svm.SVC(kernel='linear', C=C).fit(X, y) # high 60s
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y) # Work bad, 30s
        poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y) # mid 60s
        lin_svc = svm.LinearSVC(C=C).fit(X, y) # mid 60s

        model_scores = []
        for model in [svc, rbf_svc, poly_svc, lin_svc]:
            model_scores.append(validation_score(model, X_test, y_test))
        scores.append(model_scores)

    return scores

def validation_score(model, inputs, targets):
    return (model.predict(inputs) == targets).sum() / float(len(targets))

def validation(images, labels, model):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(images, labels, test_size=0.3, random_state=0)

    return model.fit(X_train, y_train).score(X_test, y_test)

if __name__ == '__main__':
    labels, ids, images = load_labeled()
    test_im = load_test()

    testim_mod = rearrange(test_im)

    #logistic_test(images, labels, testim)
    print svm_linear_validation(images, labels, ids)
