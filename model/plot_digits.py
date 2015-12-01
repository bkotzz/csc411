import numpy as np
import matplotlib.pyplot as plt

def plot_digits(digit_array, start_inc, end_exc):
    """Visualizes each example in digit_array.

    Note: N is the number of examples 
          and M is the number of features per example.

    Inputs:
        digits: N x M array of pixel intensities.
    """

    for i in xrange(start_inc, end_exc):
        plt.imshow(digit_array[:, :, i])
        plt.show()


