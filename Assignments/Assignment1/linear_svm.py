from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]  # Q. W(D, C)에서 C는 무엇인가요?? => 변수명에 num_classes
    num_train = X.shape[0]    # num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)  # X*D
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1, margin을 1 주는 것이다.
            if margin > 0:
                loss += margin

                # reference: https://github.com/jariasf/CS231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py
                dW[:, y[i]] = dW[:, y[i] - X[i]]  
                dW[:, j] = dW[:,j] + X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW = dW / num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW = dW + reg * 2 * W 
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pass
    # 여기서부터 gradient를 구하는 코드를 구현하면 된다 이거지!
    # 쉽지 않겠군... -20.04.22.wed.pm1:11-

    # Implement Gradient



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # reference: https://github.com/jariasf/CS231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py
    # compute loss
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = np.dot(X, W)
    correct_class_scores = scores[np.arange(num_train), y].reshape(num_train, 1)

    # compute margin ( scores)
    # scores에서 correct_class_scores를 뺴줘야 margin을 구할 수 있다.
    margin = np.maximum(0, scores - correct_class_scores + 1)
    margin[np.arange(num_train), y] = 0  # do not consider correct class in loss
                                         # 실제 정답인 클래스들은 마진을 0으로 한다.
    loss = margin.sum() / num_train
    
    # Add regularization to the loss
    loss += reg * np.sum(W*W)

    # pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # reference: https://github.com/MahanFathi/CS231/blob/master/assignment1/cs231n/classifiers/linear_svm.py
    # compute gradient
    # pass
    X_mask = np.zeros(margin.shape)
    X_mask[margin > 0] = 1  # margin이 0보다 크면 모두 True
    X_mask[np.arange(num_train), y] -= np.sum(X_mask, axis=1)

    #subtract in correct class (-s_y)
    dW = (X.T).dot(X_mask) / num_train

    # Regularization gradient
    dW = dW + reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
