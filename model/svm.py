import numpy as np
from plot_digits import *
from utils import *
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import LabelKFold
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV

def svm_test(X_train, y_train, X_test):

    n = 142
    X_unlabeled = load_unlabeled()
    X_train_pca, X_test_pca = perform_pca(np.vstack((X_train, X_unlabeled, X_test)), X_train, X_test, n)

    param_grid = {'C': [4.65e1, 4.8e1, 5e1, 5.2e1, 5.35e1], 'gamma': [0.00475, 0.004625, 0.0045, 0.004375, 0.00425, 0.004125] }
    model = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid).fit(X_train_pca, y_train)
    print model.best_params_
    print 'after fitting'
    test_labels = model.predict(X_test_pca)

    create_submission(test_labels)

def svm_pca_validation(images, labels, ids):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(images, labels, test_size=0.25, random_state=0)
    X_unlabeled = load_unlabeled()

    model_scores = []
    n_values = [138, 140, 142, 144, 146, 148, 150]
    for n in n_values:
        print n
        X_train_pca, X_test_pca = perform_pca(np.vstack((X_train, X_unlabeled)), X_train, X_test, n)
        print 'after pca'

        param_grid = {'C': [400], 'gamma': [0.004] }
        model = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid).fit(X_train_pca, y_train)
        print model.best_params_
        print 'after fitting'
        model_scores.append(model.score(X_test_pca, y_test))

    plt.plot(n_values, model_scores, label='Validation Set')
    plt.xlabel('Number of Components')
    plt.ylabel('Classification Rate')
    plt.legend()
    plt.title('Classification Rates for a Polynomial Kernel')
    plt.ylim([0, 1])
    plt.show()

    return model_scores

def svm_linear_validation(images, labels, ids):
    lkf = LabelKFold(ids, n_folds=2)

    model_scores = []
    C_values = [1, 10, 100, 1000, 10000]
    for C in C_values:
        model = svm.SVC(kernel='linear', C=C)
        model_scores.append(cross_validation.cross_val_score(model, images, labels, cv=lkf).max())

    plt.plot(C_values, model_scores, label='Validation Set')
    plt.xlabel('C Values')
    plt.ylabel('Classification Rate')
    plt.legend()
    plt.title('Classification Rates for a Linear Kernel')
    plt.ylim([0, 1])
    plt.xscale('log')
    plt.show()

    return model_scores

def svm_validation(images, labels, ids):

    lkf = LabelKFold(ids, n_folds=2)

    C = 1.0  # SVM regularization parameter

    svc = svm.SVC(kernel='linear', C=C) 
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C)
    lin_svc = svm.LinearSVC(C=C)

    model_scores = []
    for model in [svc, rbf_svc, poly_svc, lin_svc]:
        model_scores.append(cross_validation.cross_val_score(model, images, labels, cv=lkf).max())

    return model_scores

def validation(images, labels, model):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(images, labels, test_size=0.3, random_state=0)

    return model.fit(X_train, y_train).score(X_test, y_test)

def perform_pca(fit, transform1, transform2, n):
    model = PCA(n_components = n, whiten = True)
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
    hidden_im = load_hidden()
    total_test = np.vstack((test_im, hidden_im))


    #svm_test_pca(images, labels, test_im)
    #print svm_poly_validation(images, labels, ids)
    #print svm_linear_validation(images, labels, ids)
    #print svm_pca_validation(images, labels, ids)
    #print svm_validation(images, labels, ids)
    svm_test(images, labels, total_test)

