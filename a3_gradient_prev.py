from data_utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt

def GD(x_train, y_train, x_test, y_test, th0, eg, ea=1e-6, er=0.01):
    k = 0
    thk = th0 # weights
    f_thk = float('-inf')
    f_thkp1 = float('nan')
    succ_count = 0

    # losses
    losses = np.array([])
    # iter
    iters = np.array([])
    # "test" (validation) data losses
    # losses_test = np.array([])

    while True:
        # find gradient gk
        f_pred = _sigmoid(thk, x_train)
        gk = _grad(x_train, y_train,f_pred)
        # print(gk.shape)
        if np.linalg.norm(gk) <= eg:
            break
        else:
            pk = -1*gk/np.linalg.norm(gk)
        # line search for ak
        ak = _line_search(x=x_train, y=y_train, a_bar=0.1, gk=gk, pk=pk, thk=thk)
        print(ak)
        # update th
        thk += ak*pk
        print(thk)
        # evaluate f thk+1
        f_thkp1 = _f(x_train, y_train, thk)
        # record loss
        losses = np.append(losses,f_thkp1)
        #y_pred = np.rint(_sigmoid(thk, x_train))
        #y_actual = y_train.astype(int)
        #loss = _rmse(y_pred, y_actual)
        #losses = np.append(losses, loss)
        #y_pred_test = np.rint(_sigmoid(thk, x_test))
        #y_actual_test = y_test.astype(int)
        #loss_test = _rmse(y_pred_test, y_actual_test)
        #losses_test = np.append(losses_test, loss_test)
        iters = np.append(iters, k)
        # calculate stopping condition
        if k > 0 and np.abs(f_thkp1-f_thk)<ea+er*np.abs(f_thk):
            f_thk = f_thkp1
            succ_count+=1
            if succ_count == 2:
                break
            k+=1
        else:
            if succ_count==1:
                succ_count = 0  # reset success count
            f_thk = f_thkp1
            k+=1
    #return thk, losses, losses_test, iters
    return thk, losses, iters

def _sigmoid (th, x):
    # print("in")
    # 1/1+e^x
    exponent = np.dot(x,th)
    #print(exponent.shape)
    den = np.ones((exponent.shape[0],1)) + np.exp(exponent)
    #print(den.shape)
    result = 1./den
    #print(result.shape)
    return result

def _grad(x, y, f_pred):
    return (np.sum(np.subtract(y,f_pred)*x,axis=0)).reshape((5,1))

def _f(x,y,th):
    fHat = _sigmoid(th, x)
    # NLL of Bernoulli
    first = np.vdot(y, (np.log(fHat)))
    # print(first)
    temp1 = np.subtract(np.ones((y.shape[0], y.shape[1])),y)
    #print(temp1.shape)
    temp2 = np.log(np.subtract(np.ones((fHat.shape[0], fHat.shape[1])),fHat))
    second = np.vdot(temp1, temp2)
    # print(second)
    nll = first + second
    return -1.*nll

def _rmse(y_pred, y_actual):
    '''
    calculate the root mean squared error between the estimated y values and
        the actual y values
    param y_estimates: list of ints
    param y_valid: list of ints, actual y values
    return: float, rmse value between two lists
    '''
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
    th0 = np.random.normal(0.0,1.0,(x_train.shape[1],1))

    # calculate weights
    # weights, losses, losses_t, iters = GD(x_train, y_train, x_test, y_test, th0, eg=1e-3)
    # print(losses.shape)
    # print(losses_t.shape)
    # _plot2(iters, losses, losses_t, "Training loss", "Testing loss","Iteration #", "Loss", "Loss vs Iterations")
    weights, losses, iters = GD(x_train, y_train, x_test, y_test, th0, eg=1e-3)
    plt.plot(losses, iters)
    plt.show()