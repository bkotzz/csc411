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

    return labels.ravel(), identities.ravel(), rearrange(images)

def load_unlabeled():
    """Loads unlabeled data."""
    mat = loadmat('../unlabeled_images')
    images = mat['unlabeled_images']

    return rearrange(images)

def load_test():
    """Loads test data."""
    mat = loadmat('../public_test_images')
    images = mat['public_test_images']

    return rearrange(images)


def create_submission(prediction):
    file = open("submission.csv", "w+")
    file.write("Id,Prediction\n")

    for i in xrange(len(prediction)):
        file.write(str(i + 1) + "," + str(int(prediction[i])) + "\n")

    for i in xrange(len(prediction), 1253):
        file.write(str(i + 1) + ",0\n")

    file.close()

def rearrange(a): return a.reshape((a.shape[0] * a.shape[1], a.shape[2])).T

