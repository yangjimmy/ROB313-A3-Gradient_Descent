import autograd.numpy as np
from autograd import value_and_grad
from data_utils import load_dataset
from data_utils import plot_digit
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
    Fhat = _log_softmax(H2, W3, b3)

    return Fhat

def _softmax(x,w,b):
    z = np.dot(x, w.T) + b.T
    temp = np.exp(z)
    den = np.sum(temp)
    return z/den

def _log_softmax(x,w,b):
    z = np.dot(x, w.T) + b.T
    # find max along axis
    max_vals = np.amax(z, axis=1)
    max_vals = np.reshape(max_vals, (x.shape[0],1))
    # use log sum exp trick covered in class
    max_vals_matrix = np.repeat(max_vals, 10, axis=1)
    temp_matrix = np.exp(np.subtract(z,max_vals_matrix))
    temp = np.sum(temp_matrix, axis=1)
    temp2 = np.reshape(temp, (x.shape[0],1))
    temp3 = np.repeat(temp2, 10, axis=1)
    return np.subtract(z,(np.add(np.log(temp3),max_vals_matrix)))


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
    # NLL of Gaussian
    # nll = 0.5*np.sum(np.square(Fhat - y)) + 0.5*y.size*np.log(2.*np.pi)
    # print(nll)
    # NLL of softmax
    matrix = np.dot(Fhat, y.T)
    temp_vector = np.diag(matrix)
    return -1.*np.sum(temp_vector)


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


def _plot2(x,y1,y2,legend1,legend2,x_label,y_label,title, save=True):
    line1,line2 = plt.plot(x,y1,x,y2)
    plt.legend((line1,line2),(legend1,legend2))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if save:
        plt.savefig("{}.png".format(title))
    plt.show()
    
def run_example():
    """
    This example demonstrates computation of the negative log likelihood (nll) as
    well as the gradient of the nll with respect to all weights and biases of the
    neural network. We will use 50 neurons per hidden layer and will initialize all 
    weights and biases to zero.
    """
    # load the MNIST_small dataset
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')
    
    # initialize the weights and biases of the network
    # initialize with zeroes
    M = 50 # 50 neurons per hidden layer
    W1 = np.zeros((M, 784)) # weights of first (hidden) layer
    W2 = np.zeros((M, M)) # weights of second (hidden) layer
    W3 = np.zeros((10, M)) # weights of third (output) layer
    b1 = np.zeros((M, 1)) # biases of first (hidden) layer
    b2 = np.zeros((M, 1)) # biases of second (hidden) layer
    b3 = np.zeros((10, 1)) # biases of third (output) layer
    
    # considering the first 250 points in the training set, 
    # compute the negative log likelihood and its gradients

    negative_log_likelihood(W1, W2, W3, b1, b2, b3, x_train[:250], y_train[:250])
    (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = \
        nll_gradients(W1, W2, W3, b1, b2, b3, x_train[:250], y_train[:250])
    print("negative log likelihood: %.5f" % nll)


def sgd(x_train, y_train, x_valid, y_valid, x_test, y_test, layer_size=100, batch_size=250, learning_rate=0.001, num_iters=100, ud=5, save=True):
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
    np.random.seed(100)
    scale = 1
    W1 = scale*np.random.randn(layer_size, 784) / np.sqrt(784/2)
    W2 = scale*np.random.randn(layer_size, layer_size) / np.sqrt(layer_size/2)
    W3 = scale*np.random.randn(10, layer_size) / np.sqrt(layer_size)
    b1 = np.zeros((layer_size, 1))  # biases of first (hidden) layer
    b2 = np.zeros((layer_size, 1))  # biases of second (hidden) layer
    b3 = np.zeros((10, 1))  # biases of third (output) layer

    losses_train = np.array([])
    losses_valid = np.array([])
    iters = np.array([])

    for iter in range(num_iters):
        idxs = np.random.randint(x_train.shape[0], size=batch_size)
        (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = \
            nll_gradients(W1, W2, W3, b1, b2, b3, x_train[idxs,:], y_train[idxs,:])
        # print("negative log likelihood: %.5f" % nll)
        # record training loss
        losses_train = np.append(losses_train, nll/batch_size)

        # assume that gradients averaged already i.e. 1/|B|
        W1 -= learning_rate*W1_grad
        W2 -= learning_rate*W2_grad
        W3 -= learning_rate*W3_grad
        b1 -= learning_rate*b1_grad
        b2 -= learning_rate*b2_grad
        b3 -= learning_rate*b3_grad

        # find validation loss
        (nll2, (W1_grad2, W2_grad2, W3_grad2, b1_grad2, b2_grad2, b3_grad2)) = \
            nll_gradients(W1, W2, W3, b1, b2, b3, x_valid, y_valid)
        # loss_valid = negative_log_likelihood(W1, W2, W3, b1, b2, b3, x_valid, y_valid)
        losses_valid = np.append(losses_valid, nll2/x_valid.shape[0])
        # record iter number
        iters = np.append(iters, iter)
        nll=0
        W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad = 0,0,0,0,0,0

    (loss_test, (W1_grad2, W2_grad2, W3_grad2, b1_grad2, b2_grad2, b3_grad2)) = \
        nll_gradients(W1, W2, W3, b1, b2, b3, x_test, y_test)
    print("Average test loss: {}".format(loss_test/x_test.shape[0]))
    print("Final average training loss: {}".format(losses_train[losses_train.shape[0]-1]))
    print("Final average validation loss: {}".format(losses_valid[losses_valid.shape[0]-1]))

    _plot2(iters, losses_valid, losses_train, "Validation", "Training", "Iteration #", "Average Loss", "Multiclass Classification Average Loss vs Iterations (Learning Rate {})".format(learning_rate), save)
    x_vals = []
    if ud != 0:
        for i in range(x_test.shape[0]):
            prob = np.max(np.exp(forward_pass(W1, W2, W3, b1, b2, b3, x_test[i])))
            if prob<0.5:
                # print("probability: {}".format(prob))
                x_vals.append(x_test[i])
                axis = plt.subplot(1, ud, len(x_vals))
                axis.imshow(x_test[i].reshape((28, 28)), interpolation='none', aspect='equal', cmap='gray')
            if len(x_vals)>=ud:
                break
    if save:
        plt.savefig("{}.png".format("Unconfident Inputs"))
    plt.show()



if __name__ == '__main__':
    # run_example()
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')

    layer_size = 100 # size of hidden layers
    batch_size = 250 # batch size during training
    learning_rate = 0.0001 # learning rate
    num_iters = 200 # number of iterations to perform
    ud = 5 # number of Unconfident Data inputs to plot
    save = True # whether to save final plots
    sgd(x_train, y_train, x_valid, y_valid, x_test, y_test,layer_size=layer_size, batch_size=batch_size, learning_rate=learning_rate, num_iters=num_iters, ud=ud, save=save)

