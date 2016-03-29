import numpy as np
from check_grad import check_grad
from plot_digits import *
from utils import *
from logistic import *

def run_logistic_regression(lambda_i = 0.001): # 0.01
    #train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 1, # 0.2,
                    'weight_regularization': lambda_i,
                    'num_iterations': 30 # 50
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.randn(M + 1, 1) - 0.36

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    ce_training = np.zeros((hyperparameters['num_iterations']))
    ce_validation = np.zeros((hyperparameters['num_iterations']))

    # Begin learning with gradient descent
    for t in xrange(hyperparameters['num_iterations']):

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        
        # print some stats
        stat_msg = "ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f}  "
        stat_msg += "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}"
        print stat_msg.format(t+1,
                              float(f / N),
                              float(cross_entropy_train),
                              float(frac_correct_train*100),
                              float(cross_entropy_valid),
                              float(frac_correct_valid*100))

        ce_training[t] = cross_entropy_train
        ce_validation[t] = cross_entropy_valid

    print weights.mean()

    plt.plot(xrange(1, hyperparameters['num_iterations'] + 1), ce_training, label='Training Set')
    plt.plot(xrange(1, hyperparameters['num_iterations'] + 1), ce_validation, label='Validation Set')
    plt.xlabel('Iteration Number')
    plt.ylabel('Cross Entropy')
    plt.legend()
    plt.title('Cross Entropy by iteration number')
    plt.show()

    return cross_entropy_train, frac_correct_train, cross_entropy_valid, frac_correct_valid

def find_lambda():
    lambda_vals = [0.001, 0.01, 0.1, 1.0]

    ce_train_averages = np.zeros(4)
    cr_train_averages = np.zeros(4)
    ce_valid_averages = np.zeros(4)
    cr_valid_averages = np.zeros(4)

    for lambda_i in xrange(len(lambda_vals)):
        for i in xrange(10):
            ce_train, cr_train, ce_valid, cr_valid = run_logistic_regression(lambda_vals[lambda_i])
            ce_train_averages[lambda_i] += ce_train / 10.0
            cr_train_averages[lambda_i] += cr_train / 10.0
            ce_valid_averages[lambda_i] += ce_valid / 10.0
            cr_valid_averages[lambda_i] += cr_valid / 10.0

    print ce_train_averages
    print cr_train_averages
    print ce_valid_averages
    print cr_valid_averages
            
    plt.plot(lambda_vals, ce_train_averages, label='Training Set')
    plt.plot(lambda_vals, ce_valid_averages, label='Validation Set')
    plt.xlabel('Lambda')
    plt.ylabel('Cross Entropy')
    plt.legend()
    plt.title('Cross Entropy vs Lambda')
    plt.xscale('log')
    plt.show()

    plt.plot(lambda_vals, cr_train_averages, label='Training Set')
    plt.plot(lambda_vals, cr_valid_averages, label='Validation Set')
    plt.xlabel('Lambda')
    plt.ylabel('Classification Rate')
    plt.legend()
    plt.title('Classification Rate vs Lambda')
    plt.xscale('log')
    plt.ylim(ymax=1.05)
    plt.show()

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.round(np.random.rand(num_examples, 1), 0)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':
    run_logistic_regression()
    #find_lambda()
