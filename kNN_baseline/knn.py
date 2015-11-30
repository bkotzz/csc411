import numpy as np
from plot_digits import *
from utils import *
import matplotlib.pyplot as plt
from sklearn import neighbors


def compute_classification_rate(k_values, train_inputs, train_targets, inputs, targets):
	c_rate = []

	#print train_targets
	train_inputs_mod = train_inputs.reshape((train_inputs.shape[0] * train_inputs.shape[1], train_inputs.shape[2])).T
	inputs_mod = inputs.reshape((inputs.shape[0] * inputs.shape[1], inputs.shape[2])).T
	targets_mod = targets.ravel()
	train_targets_mod = train_targets.ravel()

	for k in k_values:

		clf = neighbors.KNeighborsClassifier(k, weights='uniform')
		clf.fit(train_inputs_mod, train_targets_mod)

		labels = clf.predict(inputs_mod)
		#print targets
		#print labels
		num_correct_labels = np.sum(targets_mod == labels)
		#print num_correct_labels
		c_rate.append(num_correct_labels / float(len(targets_mod)))

	return c_rate


if __name__ == '__main__':
	labels, ids, images = load_labeled()
	test_im = load_test()

	k_values = [1, 3, 5, 7, 9, 11, 13, 15] #, 17, 19, 21, 23, 25]
	data_split = 2000
	limit = -1
	test_c_rate  = compute_classification_rate(k_values, images[:, :, :data_split], labels[:data_split], images[:, :, data_split:limit], labels[data_split:limit])

	print test_c_rate
	
	plt.plot(k_values, test_c_rate, label='Test Set')
	plt.xlabel('K Values')
	plt.ylabel('Classification Rate')
	plt.legend()
	plt.title('Classification Rates for various choices of K')
	plt.ylim([0, 1])
	plt.show()
        
