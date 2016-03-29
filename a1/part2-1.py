import numpy as np
from plot_digits import *
from run_knn import run_knn
from utils import *
import matplotlib.pyplot as plt


def compute_classification_rate(k_values, train_inputs, train_targets, inputs, targets):
	c_rate = []

	for k in k_values:
		labels = run_knn(k, train_inputs, train_targets, inputs)
		
		num_correct_labels = 0

		for i in xrange(len(targets)):
			num_correct_labels += int(targets[i] == labels[i])

		c_rate.append(num_correct_labels / float(len(targets)))

	return c_rate


if __name__ == '__main__':
	train_inputs, train_targets = load_train()
	valid_inputs, valid_targets = load_valid()
	test_inputs, test_targets = load_test()

	k_values = [1, 3, 5, 7, 9]
	valid_c_rate = compute_classification_rate(k_values, train_inputs, train_targets, valid_inputs, valid_targets)
	test_c_rate  = compute_classification_rate(k_values, train_inputs, train_targets, test_inputs, test_targets)

	print valid_c_rate
	print test_c_rate

	plt.plot(k_values, valid_c_rate, label='Validation Set')	
	plt.plot(k_values, test_c_rate, label='Test Set')
	plt.xlabel('K Values')
	plt.ylabel('Classification Rate')
	plt.legend()
	plt.title('Classification Rates for various choices of K')
	plt.ylim([0.75, 1])
	plt.show()
        
