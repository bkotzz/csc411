import numpy as np
from plot_digits import *
from utils import *
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.cross_validation import LabelKFold
from sklearn.semi_supervised import LabelPropagation


def ss_test(images, labels, unlabeled_images, test_images):
    all_images = np.vstack((images, unlabeled_images))
    neg_ones = -np.ones((unlabeled_images.shape[0],))
    all_labels = np.concatenate((labels, neg_ones), axis = 0)

    model = LabelPropagation()
    model.fit(all_images, all_labels)

    test_labels = model.predict(test_images)
    create_submission(test_labels)

if __name__ == '__main__':
    labels, ids, images = load_labeled()
    test_im = load_test()
    unlabeled_im = load_unlabeled()

    ss_test(images, labels, unlabeled_im, test_im)
