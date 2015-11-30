import numpy as np
from l2_distance import l2_distance
from scipy.stats import mode
from sklearn import neighbors

def run_knn(k, train_data, train_labels, valid_data):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples, 
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification 
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data 
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels 
                      for the validation data.
    """

    #print train_data.shape
    #print train_labels.shape
    #print valid_data.shape
    clf = neighbors.KNeighborsClassifier(k, weights='uniform')
    clf.fit(train_data, train_labels.ravel())

    valid_labels = clf.predict(valid_data)
    #print valid_labels

    return valid_labels
