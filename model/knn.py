import numpy as np
from plot_digits import *
from utils import *
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import cross_validation

def fit(k, train_inputs, train_targets):
    clf = neighbors.KNeighborsClassifier(k, weights='uniform')
    clf.fit(train_inputs, train_targets)
    return clf

def compute_classification_rate(k_values, train_inputs, train_targets, inputs, targets):
    c_rate = []

    for k in k_values:

        score = fit(k, train_inputs, train_targets).score(inputs, targets)
        c_rate.append(score)

    return c_rate

def validation(images, labels):
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(images, labels, test_size=0.3, random_state=0)
    test_c_rate = compute_classification_rate(k_values, X_train, y_train, X_test, y_test)

    print test_c_rate
    
    plt.plot(k_values, test_c_rate, label='Validation Set')
    plt.xlabel('K Values')
    plt.ylabel('Classification Rate')
    plt.legend()
    plt.title('Classification Rates for various choices of K')
    plt.ylim([0, 1])
    plt.show()

def test(labels, images, test_im):
    test_labels = fit(9, images, labels).predict(test_im)
    create_submission(test_labels)

if __name__ == '__main__':
    labels, ids, images = load_labeled()
    test_im = load_test()

    validation(images, labels)
    #test(labels, images, test_im)

        
