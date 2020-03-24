from data_utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import time

def GD(x_train, y_train, x_test, y_test, th0, learning_rate, num_iter=500, SGD=False):
    k = 0
    thk = th0 # weights
    # losses
    losses = np.array([])
    # iter
    iters = np.array([])
    # "test" (validation) data losses
    losses_test = np.array([])
    # testing accuracies
    accuracies = np.array([])

    while num_iter>0:
        # find gradient gk
        # update the theta
        # different gradient for different methods
        if SGD:
            t = np.random.randint(0,x_train.shape[0]-1)
            x_t = x_train[t,:]
            f_pred_t = _sigmoid(thk, x_t)
            y_t = y_train[t]
            grad = _grad(x_t, y_t, f_pred_t)
        else:
            f_pred = _sigmoid(thk, x_train)
            grad = _grad(x_train, y_train, f_pred)
        thk -= learning_rate*grad
        # evaluate f thk+1
        f_thkp1 = _f(x_train, y_train, thk)
        # record loss
        losses = np.append(losses,f_thkp1)
        losses_test = np.append(losses_test, _f(x_test, y_test, thk))
        y_pred_test = _sigmoid(thk, x_test)
        accuracies = np.append(accuracies, _accuracy(y_test, y_pred_test))
        iters = np.append(iters, k)
        # calculate stopping condition
        k += 1
        num_iter -= 1

    return thk, losses, losses_test, accuracies, iters


def run_GD(x_train, y_train, x_test, y_test, lr_desired, num_iter, save=True):
    for lr in lr_desired:
        curr = time.time()
        th0 = np.zeros((x_train.shape[1],1))
        thk, losses, losses_test, accuracies, iters = GD(x_train, y_train, x_test, y_test, th0, learning_rate=lr, num_iter = num_iter)
        end = time.time()
        plt.plot(losses, label="Learning rate {}".format(lr))
        print("Time elapsed: {}".format(end - curr))
        print("Full Batch GD Training loss: {}".format(losses[losses.shape[0] - 1]))
        print("Full Batch GD Testing accuracy: {}, Loss: {}; learning rate = {}".format(accuracies[accuracies.shape[0]-1], losses_test[losses_test.shape[0]-1], lr))
    plt.legend(loc="upper right")
    plt.xlabel("Iteration #")
    plt.ylabel("Negative Log Likelihood (Loss)")
    plt.title("Full Batch GD Loss vs Iteration #")
    if save:
        plt.savefig("Full Batch GD.png")
    plt.show()


def run_SGD(x_train, y_train, x_test, y_test, lr_desired, num_iter, save=True):
    for lr in lr_desired:
        curr = time.time()
        th0 = np.zeros((x_train.shape[1],1))
        thk, losses, losses_test, accuracies, iters = GD(x_train, y_train, x_test, y_test, th0, learning_rate=lr, num_iter = num_iter, SGD=True)
        end = time.time()
        plt.plot(losses, label="Learning rate {}".format(lr))
        print("Time elapsed: {}".format(end-curr))
        print("SGD Training loss: {}".format(losses[losses.shape[0]-1]))
        print("SGD Testing accuracy: {}, Loss: {}; learning rate = {}".format(accuracies[accuracies.shape[0]-1], losses_test[losses_test.shape[0]-1],lr))
    plt.legend(loc="upper right")
    plt.xlabel("Iteration #")
    plt.ylabel("Negative Log Likelihood (Loss)")
    plt.title("SGD Loss vs Iteration #")
    if save:
        plt.savefig("SGD.png")
    plt.show()



def _sigmoid (th, x):
    # 1/1+e^z
    exponent = np.dot(x,th)
    den = np.ones((exponent.shape[0],1)) + np.exp(exponent)
    result = 1./den
    return result

def _grad(x, y, f_pred):
    return (np.sum(np.subtract(y,f_pred)*x,axis=0)).reshape((5,1))

def _f(x,y,th):
    fHat = _sigmoid(th, x)
    # NLL of Bernoulli
    first = np.vdot(y, (np.log(fHat)))
    temp1 = np.subtract(np.ones((y.shape[0], y.shape[1])),y)
    temp2 = np.log(np.subtract(np.ones((fHat.shape[0], fHat.shape[1])),fHat))
    second = np.vdot(temp1, temp2)
    nll = first + second
    return -1.*nll

def _rmse(y_pred, y_actual):
    """
    calculate the root mean squared error between the estimated y values and
    the actual y values
    :param y_estimates: list of ints
    :param y_valid: list of ints, actual y values
    :return: float, rmse value between two lists
    """
    return np.sqrt(np.average(np.abs(y_pred-y_actual)**2))

def _line_search(x, y, a_bar, gk, pk, thk, m1 = 1e-4, r=0.5):
    a = a_bar
    # compute f thk
    f_thk = _f(x,y,thk)
    f_thk_apk = _f(x,y,thk+a*pk)
    # iterate
    while f_thk_apk > (f_thk + m1*a*np.vdot(gk,pk)):
        a *= r
        f_thk_apk = _f(x, y, thk + a * pk)
    return a

def _accuracy(y,y_pred):
    y_pred=np.rint(y_pred)
    result = 0
    for i in range(y_pred.shape[0]):
        if y[i]==y_pred[i]:
            result+=1
    return result*1.0/y_pred.shape[0]

def _plot2(x,y1,y2,legend1,legend2,x_label,y_label,title):
    line1,line2 = plt.plot(x,y1,x,y2)
    plt.legend((line1,line2),(legend1,legend2))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    y_train = y_train[:,(1,)]
    y_valid = y_valid[:,(1,)]
    y_test = y_test[:,(1,)]

    # merge training and testing into one
    x_train = np.vstack((x_train,x_valid))
    y_train = np.vstack((y_train,y_valid))

    # augment x
    x_train = np.hstack((np.ones((x_train.shape[0],1)),x_train))
    x_test = np.hstack((np.ones((x_test.shape[0],1)),x_test))

    # define all constants and values
    #th0 = np.random.normal(0.0,1.0,(x_train.shape[1],1))
    th0 = np.zeros((x_train.shape[1],1))

    # calculate weights
    # weights, losses, losses_t, accuracies, iters = GD(x_train, y_train, x_test, y_test, th0, learning_rate=0.01, num_iter=500, SGD=False)
    # plt.plot(losses)
    #
    # th0 = np.zeros((x_train.shape[1], 1))
    # weights2, losses2, losses_t2, accuracies2, iters2 = GD(x_train, y_train, x_test, y_test, th0, learning_rate=0.001, num_iter=500, SGD=False)
    # plt.plot(losses2)
    #
    # plt.show()
    #_plot2(iters, losses, losses_t, "Training loss", "Testing loss", "Iteration #", "Loss", "Loss vs Iterations")
    # print(losses[losses.shape[0]-1])
    # print(losses_t[losses_t.shape[0]-1])
    # print(accuracies[accuracies.shape[0]-1])
    # plt.show()

    lr_list = [0.01,0.001,0.0001]
    run_GD(x_train, y_train, x_test, y_test, lr_list, 3000)
    run_SGD(x_train, y_train, x_test, y_test, lr_list, 3000)