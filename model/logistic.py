import numpy as np
from plot_digits import *
from utils import *
import matplotlib.pyplot as plt
from sklearn import linear_model

def ridge_validation(images, labels):
 #   c_rate = []

 #   for alpha in np.arange(0.001,1,0.005):
 #       logistic = linear_model.Ridge(alpha)
 #       c_rate.append(validation(images, labels, logistic))

    model = linear_model.Ridge(alpha=0.1)
    return validation(images, labels, model)

def logistic_test(images, labels, test_images):
    logistic = linear_model.LogisticRegression()

    test_labels = test(images, labels, test_images, logistic)
    create_submission(test_labels)

def test(images, labels, test_images, model):
    return model.fit(images, labels).predict(test_images)

def logistic_validation(images, labels):
    logistic = linear_model.LogisticRegression()

    return validation(images, labels, logistic)

def validation(images, labels, model):
    n_samples = len(labels)

    X_train = images_mod[:0.9 * n_samples]
    y_train = labels_mod[:0.9 * n_samples]

    X_test = images_mod[0.9 * n_samples:]
    y_test = labels_mod[0.9 * n_samples:]

    return model.fit(X_train, y_train).score(X_test, y_test)

if __name__ == '__main__':
    labels, ids, images = load_labeled()
    test_im = load_test()

    images_mod = rearrange(images)
    labels_mod = labels.ravel()
    testim_mod = rearrange(test_im)

    logistic_test(images_mod, labels_mod, testim_mod)
    #print logistic_validation(images, labels)
    #print ridge_validation(images_mod, labels_mod)