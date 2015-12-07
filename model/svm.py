import numpy as np
from plot_digits import *
from utils import *
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import LabelKFold
from sklearn.decomposition import PCA

def svm_test(images, labels, test_images):

    print 'after pca'
    #model = svm.SVC(kernel='rbf', gamma=0.7, C=1).fit(images, labels)
    model = svm.SVC(kernel='poly', degree=3, C=1).fit(images, labels)
    print 'after svm fit'
    test_labels = model.predict(test_images)
    print test_labels
    print model.score(images, labels)
    print 'after svm predict'
    create_submission(test_labels)

def svm_pca_validation(images, labels, ids):
    lkf = LabelKFold(ids, n_folds=2)
    x_unlabeled = load_unlabeled()

    scores = []
    for train, test in lkf:
        y = labels[train]
        y_test = labels[test]

        poly_model_scores = []
        for n in [10, 30, 70, 120, 500]:
            print n
            X, X_test = perform_pca(x_unlabeled, images[train], images[test], n)
            print 'after pca'
            poly = svm.SVC(kernel='rbf', gamma=0.7, C=1).fit(X, y)
            print 'after fitting'
            poly_model_scores.append(poly.score(X_test, y_test))
        scores.append(poly_model_scores)

    return scores

def svm_poly_validation(images, labels, ids):
    lkf = LabelKFold(ids, n_folds=2)

    scores = []
    for train, test in lkf:
        X = images[train]
        y = labels[train]

        X_test = images[test]
        y_test = labels[test]

        model_scores = []
        for C in [1, 10, 100, 1000, 10000]:
            model = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y) # high 60s for all C
            model_scores.append(model.score(X_test, y_test))
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

def validation(images, labels, model):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(images, labels, test_size=0.3, random_state=0)

    return model.fit(X_train, y_train).score(X_test, y_test)

def perform_pca(fit, transform1, transform2, n):
    model = PCA(n_components = n, whiten = False)
    model.fit(fit)

    return model.transform(transform1), model.transform(transform2)

def svm_test_pca(x_labeled, y_labeled, x_test):
    x_unlabeled = load_unlabeled()

    n_comp = 100
    x_labeled_t, x_test_t = perform_pca(x_unlabeled, x_labeled, x_test, n_comp)

    svm_test(x_labeled_t, y_labeled, x_test_t)

if __name__ == '__main__':
    labels, ids, images = load_labeled()
    test_im = load_test()

    #svm_test_pca(images, labels, test_im)
    #print svm_poly_validation(images, labels, ids)
    print svm_pca_validation(images, labels, ids)
