import numpy as np
from plot_digits import *
from utils import *
import matplotlib.pyplot as plt
from sklearn import neighbors

def rearrange(a): return a.reshape((a.shape[0] * a.shape[1], a.shape[2])).T

def make_prediction(k, train_inputs, train_targets, inputs):
    clf = neighbors.KNeighborsClassifier(k, weights='uniform')
    clf.fit(train_inputs, train_targets)
    return clf.predict(inputs)

def compute_classification_rate(k_values, train_inputs, train_targets, inputs, targets):
    c_rate = []

    #print train_targets
    train_inputs_mod = rearrange(train_inputs)
    inputs_mod = rearrange(inputs)
    targets_mod = targets.ravel()
    train_targets_mod = train_targets.ravel()

    best_label_rate = 0

    for k in k_values:

        labels = make_prediction(k, train_inputs_mod, train_targets_mod, inputs_mod)
        #print targets
        #print labels
        num_correct_labels = np.sum(targets_mod == labels)
        #print num_correct_labels
        c_rate.append(num_correct_labels / float(len(targets_mod)))

        if c_rate[-1] > best_label_rate:
            best_labels = labels

    return c_rate, best_labels

def validation(images, labels):
    k_values = [1, 3, 5, 7, 9, 11, 13, 15] #, 17, 19, 21, 23, 25]
    data_split = 2800
    limit = -1
    test_c_rate, best_labels  = compute_classification_rate(k_values, images[:, :, :data_split], labels[:data_split], images[:, :, data_split:limit], labels[data_split:limit])
    #create_submission(best_labels)

    print test_c_rate
    
    plt.plot(k_values, test_c_rate, label='Test Set')
    plt.xlabel('K Values')
    plt.ylabel('Classification Rate')
    plt.legend()
    plt.title('Classification Rates for various choices of K')
    plt.ylim([0, 1])
    plt.show()

def test(labels, images, test_im):
    test_labels = make_prediction(5, rearrange(images), labels.ravel(), rearrange(test_im))
    create_submission(test_labels)

if __name__ == '__main__':
    labels, ids, images = load_labeled()
    test_im = load_test()

    #validation(images, labels)
    test(labels, images, test_im)


        
