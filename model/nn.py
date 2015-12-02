from utils import *
import sys
import matplotlib.pyplot as plt
from sklearn import cross_validation
plt.ion()

def InitNN(num_inputs, num_hiddens, num_outputs):
  """Initializes NN parameters."""
  W1 = 0.0001 * np.random.randn(num_inputs, num_hiddens)
  W2 = 0.0001 * np.random.randn(num_hiddens, num_outputs)
  b1 = np.zeros((num_hiddens, 1))
  b2 = np.zeros((num_outputs, 1))
  return W1, W2, b1, b2

def LoadData():
  labels, ids, images = load_labeled()

  X_train, X_test, y_train, y_test = cross_validation.train_test_split(images, labels, test_size=0.3, random_state=0)

  return X_train.T, X_test.T, y_train.T, y_test.T

def TrainNN(num_hiddens, eps, momentum, num_epochs, inputs_train, target_train, inputs_valid, index):
  """Trains a single hidden layer NN.

  Inputs:
    num_hiddens: NUmber of hidden units.
    eps: Learning rate.
    momentum: Momentum.
    num_epochs: Number of epochs to run training for.

  Returns:
    W1: First layer weights.
    W2: Second layer weights.
    b1: Hidden layer bias.
    b2: Output layer bias.
    train_error: Training error at at epoch.
    valid_error: Validation error at at epoch.
  """

  target_train = (target_train == index) * 1

  W1, W2, b1, b2 = InitNN(inputs_train.shape[0], num_hiddens, target_train.shape[0])
  dW1 = np.zeros(W1.shape)
  dW2 = np.zeros(W2.shape)
  db1 = np.zeros(b1.shape)
  db2 = np.zeros(b2.shape)
  train_error_ce = []
  train_error_rate = []

  num_train_cases = inputs_train.shape[1]
  for epoch in xrange(num_epochs):
    # Forward prop
    h_input = np.dot(W1.T, inputs_train) + b1  # Input to hidden layer.
    h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
    logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
    prediction = 1 / (1 + np.exp(-logit))  # Output prediction.

    # Compute cross entropy
    pred_result = np.abs(prediction - target_train)

    train_rate = np.where(pred_result > 0.5)[1].size / float(prediction.size)
    train_CE = -np.mean(target_train * np.log(prediction) + (1 - target_train) * np.log(1 - prediction))

    # Compute deriv
    dEbydlogit = prediction - target_train

    # Backprop
    dEbydh_output = np.dot(W2, dEbydlogit)
    dEbydh_input = dEbydh_output * h_output * (1 - h_output)

    # Gradients for weights and biases.
    dEbydW2 = np.dot(h_output, dEbydlogit.T)
    dEbydb2 = np.sum(dEbydlogit, axis=1).reshape(-1, 1)
    dEbydW1 = np.dot(inputs_train, dEbydh_input.T)
    dEbydb1 = np.sum(dEbydh_input, axis=1).reshape(-1, 1)

    #%%%% Update the weights at the end of the epoch %%%%%%
    dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1
    dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2
    db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1
    db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2

    W1 = W1 + dW1
    W2 = W2 + dW2
    b1 = b1 + db1
    b2 = b2 + db2

    train_error_ce.append(train_CE)
    train_error_rate.append(train_rate)

    sys.stdout.write('\rStep %d Train CE %.5f' % (epoch, train_CE))
    sys.stdout.flush()
    if (epoch % 100 == 0):
      sys.stdout.write('\n')

  sys.stdout.write('\n')

  final_train_error, final_train_rate = Evaluate(inputs_train, target_train, W1, W2, b1, b2)
  print 'Error: Train %.5f ' % (final_train_error)
  print 'Misclassification Rate: Train %.5f ' % (final_train_rate)
  return MakePred(inputs_valid, W1, W2, b1, b2), train_error_ce, train_error_rate

def Evaluate(inputs, target, W1, W2, b1, b2):
  """Evaluates the model on inputs and target."""

  h_input = np.dot(W1.T, inputs) + b1  # Input to hidden layer.
  h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
  logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
  prediction = 1 / (1 + np.exp(-logit))  # Output prediction.

  pred_result = np.abs(prediction - target)

  class_error = np.where(pred_result > 0.5)[1].size / float(prediction.size)

  CE = -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))

  return CE, class_error

def MakePred(inputs, W1, W2, b1, b2):
  """Evaluates the model on inputs and target."""

  h_input = np.dot(W1.T, inputs) + b1  # Input to hidden layer.
  h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
  logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
  prediction = 1 / (1 + np.exp(-logit))  # Output prediction.

  return prediction

def DisplayErrorPlot(train_error, valid_error, y_label):
  plt.figure(1)
  plt.clf()
  plt.plot(range(len(train_error)), train_error, 'b', label='Train')
  plt.plot(range(len(valid_error)), valid_error, 'g', label='Validation')
  plt.xlabel('Epochs')
  plt.ylabel(y_label)
  plt.legend()
  plt.draw()
  raw_input('Press Enter to exit.')

def SaveModel(modelfile, W1, W2, b1, b2, train_error, valid_error):
  """Saves the model to a numpy file."""
  model = {'W1': W1, 'W2' : W2, 'b1' : b1, 'b2' : b2,
           'train_error' : train_error, 'valid_error' : valid_error}
  print 'Writing model to %s' % modelfile
  np.savez(modelfile, **model)

def LoadModel(modelfile):
  """Loads model from numpy file."""
  model = np.load(modelfile)
  return model['W1'], model['W2'], model['b1'], model['b2'], model['train_error'], model['valid_error']

def MultiClassPred(preds, target):
  target = target - 1

  N = target.size

  num_correct = 0
  maximums = preds.max(axis = 0)

  for i in xrange(N):
    num_correct += (preds[target[i]][i] == maximums[i])

  return num_correct / float(N)

def RunKnn():
  num_hiddens = 10
  eps = 0.01
  momentum = 0
  num_epochs = 30

  X_train, X_test, y_train, y_test = LoadData()

  preds = np.ndarray(shape = (0, y_test.size))
  for index in [1, 2, 3, 4, 5, 6, 7]:
    print 'Model ', index
    pred, train_error_ce, train_error_rate = TrainNN(num_hiddens, eps, momentum, num_epochs, X_train, y_train, X_test, index)
    pred_mean = pred.mean(axis = 0)
    preds = np.vstack((preds, pred_mean)) 

  return preds, y_test

def main():
  preds, y_test = RunKnn()

  print 'Overall rate: ', MultiClassPred(preds, y_test)

  ##DisplayErrorPlot(train_error_ce, valid_error_ce, 'Cross Entropy')
  ##DisplayErrorPlot(train_error_rate, valid_error_rate, 'Classification Error Rate')
  # If you wish to save the model for future use :
  # outputfile = 'model.npz'
  # SaveModel(outputfile, W1, W2, b1, b2, train_error, valid_error)

if __name__ == '__main__':
  main()