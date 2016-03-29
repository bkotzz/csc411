import numpy as np
from plot_digits import *
from utils import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.cross_validation import LabelKFold

def ridge_validation(images, labels):
 #   c_rate = []

 #   for alpha in np.arange(0.001,1,0.005):
 #       logistic = linear_model.Ridge(alpha)
 #       c_rate.append(validation(images, labels, logistic))

    model = linear_model.Ridge(alpha=0.1)
    return validation(images, labels, model)

def logistic_test(images, labels, test_images):
    logistic = linear_model.LogisticRegression()

    test_labels = logistic.fit(images, labels).predict(test_images)
    create_submission(test_labels)

def logistic_validation(images, labels, ids):

    l1_scores = []
    l2_scores = []
    C_values = [0.01, 0.1, 1, 10]

    for C in C_values:
        l1 = linear_model.LogisticRegression(C=C, penalty='l1')
        l2 = linear_model.LogisticRegression(C=C, penalty='l2')
        l1_scores.append(validation_score(images, labels, l1, ids).max())
        l2_scores.append(validation_score(images, labels, l2, ids).max())
    
    plt.plot(C_values, l1_scores, label='L1 Regularization')
    plt.plot(C_values, l2_scores, label='L2 Regularization')
    plt.xlabel('C Values')
    plt.ylabel('Classification Rate')
    plt.legend()
    plt.title('LR Classification Rates for various regularizations')
    plt.ylim([0, 1])
    plt.xscale('log')
    plt.show()

    return l1_scores, l2_scores

def validation_score(images, labels, model, ids):
    lkf = LabelKFold(ids, n_folds=2) # 71% with 10 folds, 68% with 2 folds

    return cross_validation.cross_val_score(model, images, labels, cv=lkf) # cv=3 gets 61%

def validation(images, labels, model):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(images, labels, test_size=0.3, random_state=0)

    return model.fit(X_train, y_train).score(X_test, y_test)

if __name__ == '__main__':
    labels, ids, images = load_labeled()
    test_im = load_test()

    #logistic_test(images, labels, test_im)
    print logistic_validation(images, labels, ids)
    #print ridge_validation(images, labels)