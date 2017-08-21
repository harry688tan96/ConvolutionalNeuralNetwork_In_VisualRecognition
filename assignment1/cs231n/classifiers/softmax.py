import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]

  for i in xrange(N):
    Xi_dot_W = X[i].dot(W)
    Xi_dot_W -= max(Xi_dot_W) # normalization trick to prevent large intermediate terms due to exponential
    exp_func = np.exp(Xi_dot_W)

    loss += - np.log(exp_func[y[i]] / sum(exp_func))

    for j in xrange(C):
      dW[:,j] += exp_func[j] / sum(exp_func) * X[i]
      if j == y[i]:
        dW[:,j] += -X[i]

  loss = 1/N * loss + 0.5*reg*np.sum(W**2)
  dW = 1/N * dW + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]

  X_dot_W = X.dot(W)
  X_dot_W -= np.max(X_dot_W, axis=1, keepdims=True) # normalization trick to prevent large intermediate terms due to exponential
  exp_func = np.exp(X_dot_W)
  
  softmax_scores_matrix = exp_func / np.sum(exp_func, axis=1, keepdims=True)
  softmax_scores_vec = - np.log(softmax_scores_matrix[xrange(N), y])

  loss = 1/N * np.sum(softmax_scores_vec) + 0.5*reg*np.sum(W**2)

  softmax_scores = softmax_scores_matrix.copy()
  softmax_scores[xrange(N), y] += -1
  
  dW = (X.T).dot(softmax_scores)
  dW =  1/N * dW + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

