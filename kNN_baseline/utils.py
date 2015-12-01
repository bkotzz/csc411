import numpy as np
from scipy.io import loadmat

def sigmoid(x):
    """Computes the element wise logistic sigmoid of x.

    Inputs:
        x: Either a row vector or a column vector.
    """
    return 1.0 / (1.0 + np.exp(-x))

def load_labeled():
    """Loads labeled data."""
    mat = loadmat('../labeled_images')
    labels = mat['tr_labels']
    identities = mat['tr_identity']
    images = mat['tr_images']

    return labels, identities, images

def load_unlabeled():
    """Loads unlabeled data."""
    mat = loadmat('../unlabeled_images')
    images = mat['unlabeled_images']

    return images

def load_test():
    """Loads test data."""
    mat = loadmat('../public_test_images')
    images = mat['public_test_images']

    return images


def create_submission(prediction):
    file = open("submission.dat", "w+")
    file.write("Id,Prediction")

    for i in xrange(len(prediction)):
        file.write(str(i) + "," + str(prediction[i]))

    file.close()
    
