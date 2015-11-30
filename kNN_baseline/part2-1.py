import numpy as np
from plot_digits import *
from run_knn import run_knn
from utils import *
import matplotlib.pyplot as plt


def compute_classification_rate(k_values, train_inputs, train_targets, inputs, targets):
	c_rate = []

	#print train_targets

	for k in k_values:
		train_inputs_mod = train_inputs.reshape((train_inputs.shape[0] * train_inputs.shape[1], train_inputs.shape[2])).T
		inputs_mod = inputs.reshape((inputs.shape[0] * inputs.shape[1], inputs.shape[2])).T

		print train_inputs.shape
		print train_inputs_mod.shape
		labels = run_knn(k, train_inputs_mod, train_targets, inputs_mod)

		num_correct_labels = 0

		for i in xrange(len(targets)):
			num_correct_labels += int(targets[i] == labels[i])

		c_rate.append(num_correct_labels / float(len(targets)))

	return c_rate


if __name__ == '__main__':
	labels, ids, images = load_labeled()
	test_im = load_test()

	k_values = [5] #[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
	data_split = 1000
	limit = 2000
	test_c_rate  = compute_classification_rate(k_values, images[:, :, :data_split], labels[:data_split], images[:, :, data_split:limit], labels[data_split:limit])

	print test_c_rate
	
	plt.plot(k_values, test_c_rate, label='Test Set')
	plt.xlabel('K Values')
	plt.ylabel('Classification Rate')
	plt.legend()
	plt.title('Classification Rates for various choices of K')
	plt.ylim([0, 1])
	plt.show()
        
