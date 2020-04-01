import autograd.numpy as np
from autograd import value_and_grad
from data_utils import load_dataset
import matplotlib.pyplot as plt

def forward_pass(W1, W2, W3, b1, b2, b3, x):
    """
    forward-pass for an fully connected neural network with 2 hidden layers of M neurons
    Inputs:
        W1 : (M, 784) weights of first (hidden) layer
        W2 : (M, M) weights of second (hidden) layer
        W3 : (10, M) weights of third (output) layer
        b1 : (M, 1) biases of first (hidden) layer
        b2 : (M, 1) biases of second (hidden) layer
        b3 : (10, 1) biases of third (output) layer
        x : (N, 784) training inputs
    Outputs:
        Fhat : (N, 10) output of the neural network at training inputs
    """
    H1 = np.maximum(0, np.dot(x, W1.T) + b1.T) # layer 1 neurons with ReLU activation, shape (N, M)
    H2 = np.maximum(0, np.dot(H1, W2.T) + b2.T) # layer 2 neurons with ReLU activation, shape (N, M)
    Fhat = np.dot(H2, W3.T) + b3.T # layer 3 (output) neurons with linear activation, shape (N, 10)
    # #######
    # Note that the activation function at the output layer is linear!
    # You must impliment a stable log-softmax activation function at the ouput layer
    # #######
    return Fhat


def negative_log_likelihood(W1, W2, W3, b1, b2, b3, x, y):
    """
    computes the negative log likelihood of the model `forward_pass`
    Inputs:
        W1, W2, W3, b1, b2, b3, x : same as `forward_pass`
        y : (N, 10) training responses
    Outputs:
        nll : negative log likelihood
    """
    Fhat = forward_pass(W1, W2, W3, b1, b2, b3, x)
    # ########
    # Note that this function assumes a Gaussian likelihood (with variance 1)
    # You must modify this function to consider a categorical (generalized Bernoulli) likelihood
    # ########
    nll = 0.5*np.sum(np.square(Fhat - y)) + 0.5*y.size*np.log(2.*np.pi) 
    return nll
    

nll_gradients = value_and_grad(negative_log_likelihood, argnum=[0,1,2,3,4,5])
"""
    returns the output of `negative_log_likelihood` as well as the gradient of the 
    output with respect to all weights and biases
    Inputs:
        same as negative_log_likelihood (W1, W2, W3, b1, b2, b3, x, y)
    Outputs: (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad))
        nll : output of `negative_log_likelihood`
        W1_grad : (M, 784) gradient of the nll with respect to the weights of first (hidden) layer
        W2_grad : (M, M) gradient of the nll with respect to the weights of second (hidden) layer
        W3_grad : (10, M) gradient of the nll with respect to the weights of third (output) layer
        b1_grad : (M, 1) gradient of the nll with respect to the biases of first (hidden) layer
        b2_grad : (M, 1) gradient of the nll with respect to the biases of second (hidden) layer
        b3_grad : (10, 1) gradient of the nll with respect to the biases of third (output) layer
     """

    
def run_example():
    """
    This example demonstrates computation of the negative log likelihood (nll) as
    well as the gradient of the nll with respect to all weights and biases of the
    neural network. We will use 50 neurons per hidden layer and will initialize all 
    weights and biases to zero.
    """
    # load the MNIST_small dataset
    from data_utils import load_dataset
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')
    
    # initialize the weights and biases of the network
    M = 50 # 50 neurons per hidden layer
    W1 = np.zeros((M, 784)) # weights of first (hidden) layer
    W2 = np.zeros((M, M)) # weights of second (hidden) layer
    W3 = np.zeros((10, M)) # weights of third (output) layer
    b1 = np.zeros((M, 1)) # biases of first (hidden) layer
    b2 = np.zeros((M, 1)) # biases of second (hidden) layer
    b3 = np.zeros((10, 1)) # biases of third (output) layer
    
    # considering the first 250 points in the training set, 
    # compute the negative log likelihood and its gradients
    (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = \
        nll_gradients(W1, W2, W3, b1, b2, b3, x_train[:250], y_train[:250])
    print("negative log likelihood: %.5f" % nll)
    
def sgd(x_train, y_train, x_valid, y_valid, x_test, y_test, layer_size=100, batch_size=20, learning_rate=0.001, num_iters=500):
    """

    :param x_train:
    :param y_train:
    :param x_valid:
    :param y_valid:
    :param x_test:
    :param y_test:
    :param layer_size:
    :param batch_size:
    :param learning_rate: learning rate for SGD
    :param num_iters: number of iterations before stopping
    :return:
    """
    iter_count = 0

    W1 = np.zeros((layer_size, 784))  # weights of first (hidden) layer
    W2 = np.zeros((layer_size, layer_size))  # weights of second (hidden) layer
    W3 = np.zeros((10, layer_size))  # weights of third (output) layer
    b1 = np.zeros((layer_size, 1))  # biases of first (hidden) layer
    b2 = np.zeros((layer_size, 1))  # biases of second (hidden) layer
    b3 = np.zeros((10, 1))  # biases of third (output) layer

    losses_train = np.array([])
    losses_valid = np.array([])
    iters = np.array([])

    while num_iters > 0:
        iter_count += 1
        num_iters -= 1

        (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = \
            nll_gradients(W1, W2, W3, b1, b2, b3, x_train[:batch_size], y_train[:batch_size])
        print("negative log likelihood: %.5f" % nll)
        # record training loss
        losses_train = np.append(losses_train, nll)

        # assume that gradients averaged already i.e. 1/|B|
        W1 -= learning_rate*W1_grad
        W2 -= learning_rate*W2_grad
        W3 -= learning_rate*W3_grad
        b1 -= learning_rate*b1_grad
        b2 -= learning_rate*b2_grad
        b3 -= learning_rate*b3_grad

        # find validation loss
        (nll2, (W1_grad2, W2_grad2, W3_grad2, b1_grad2, b2_grad2, b3_grad2)) = \
            nll_gradients(W1, W2, W3, b1, b2, b3, x_valid[:batch_size], y_valid[:batch_size])
        # loss_valid = negative_log_likelihood(W1, W2, W3, b1, b2, b3, x_valid, y_valid)
        losses_valid = np.append(losses_valid, nll2)

        # record iter number
        iters = np.append(iters, iter_count)

    plt.plot(iters, losses_valid)
    plt.plot(iters, losses_train)
    plt.show()

def sgd2(x_train, y_train, x_valid, y_valid, x_test, y_test, layer_size=100, batch_size=250, learning_rate=0.0001, num_iters=500):
    """

    :param x_train:
    :param y_train:
    :param x_valid:
    :param y_valid:
    :param x_test:
    :param y_test:
    :param layer_size:
    :param batch_size:
    :param learning_rate: learning rate for SGD
    :param num_iters: number of iterations before stopping
    :return:
    """
    #W1 = np.random.normal(0.0, 1.0, (layer_size, 784))
    #W2 = np.random.normal(0.0, 1.0, (layer_size, layer_size))
    #W3 = np.random.normal(0.0, 1.0, (10, layer_size))
    scale = 0.1
    W1 = scale*np.random.randn(layer_size, 784) / np.sqrt(784/2)
    W2 = scale*np.random.randn(layer_size, layer_size) / np.sqrt(layer_size/2)
    W3 = scale*np.random.randn(10, layer_size)
    #W1 = np.zeros((layer_size, 784))  # weights of first (hidden) layer
    #W2 = np.zeros((layer_size, layer_size))  # weights of second (hidden) layer
    #W3 = np.zeros((10, layer_size))  # weights of third (output) layer
    b1 = scale*np.zeros((layer_size, 1))  # biases of first (hidden) layer
    b2 = scale*np.zeros((layer_size, 1))  # biases of second (hidden) layer
    b3 = scale*np.zeros((10, 1))  # biases of third (output) layer

    losses_train = np.array([])
    losses_valid = np.array([])
    iters = np.array([])

    for iter in range(num_iters):
        idxs = np.random.randint(x_train.shape[0], size=batch_size)
        (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = \
            nll_gradients(W1, W2, W3, b1, b2, b3, x_train[idxs,:], y_train[idxs,:])
        print("negative log likelihood: %.5f" % nll)
        # record training loss
        losses_train = np.append(losses_train, nll)

        # assume that gradients averaged already i.e. 1/|B|
        W1 -= learning_rate*W1_grad
        W2 -= learning_rate*W2_grad
        W3 -= learning_rate*W3_grad
        b1 -= learning_rate*b1_grad
        b2 -= learning_rate*b2_grad
        b3 -= learning_rate*b3_grad

        idxs = np.random.randint(x_valid.shape[0], size=batch_size)
        print(x_valid.shape)
        # find validation loss
        (nll2, (W1_grad2, W2_grad2, W3_grad2, b1_grad2, b2_grad2, b3_grad2)) = \
            nll_gradients(W1, W2, W3, b1, b2, b3, x_valid[idxs,:], y_valid[idxs,:])
        # loss_valid = negative_log_likelihood(W1, W2, W3, b1, b2, b3, x_valid, y_valid)
        losses_valid = np.append(losses_valid, nll2)
        # record iter number
        iters = np.append(iters, iter)
        nll=0
        W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad = 0,0,0,0,0,0

    plt.plot(losses_valid[10:], label="Validation")
    plt.plot(losses_train[10:], label="Training")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == '__main__':
    run_example()
    #x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')
    #sgd2(x_train, y_train, x_valid, y_valid, x_test, y_test)