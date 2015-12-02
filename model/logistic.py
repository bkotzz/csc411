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

    test_labels = test(images, labels, test_images, logistic)
    create_submission(test_labels)

def test(images, labels, test_images, model):
    return model.fit(images, labels).predict(test_images)

def logistic_validation(images, labels, ids):
    logistic = linear_model.LogisticRegression()

    return validation_score(images, labels, logistic, ids)

def validation_score(images, labels, model, ids):
    lkf = LabelKFold(ids, n_folds=2) # 71% with 10 folds

    return cross_validation.cross_val_score(model, images, labels, cv=lkf) # 3 gets 61%

def validation(images, labels, model):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(images, labels, test_size=0.3, random_state=0)

    return model.fit(X_train, y_train).score(X_test, y_test)

if __name__ == '__main__':
    labels, ids, images = load_labeled()
    test_im = load_test()

    testim_mod = rearrange(test_im)

    #logistic_test(images_mod, labels_mod, testim_mod)
    print logistic_validation(rearrange(images), labels.ravel(), ids.ravel())
    #print ridge_validation(images_mod, labels_mod)