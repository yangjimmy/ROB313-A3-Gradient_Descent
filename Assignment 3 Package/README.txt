a3_gradient:
the following parameters can be modified in if __name__ == '__main__':
lr_list = learning rates to try
num_iter = number of iterations
save = whether to save the images to disk

Note that running the program results in running both the gradient descent and the SGD methods. Full batch gradient descent runs in the method def run_GD while SGD runs in the method def run_SGD

a3_mod.py
the following parameters can be modified in if __name__ == '__main__':
layer_size = size of hidden layers
batch_size = batch size during training
learning_rate = learning rate
num_iters = number of iterations to perform
ud = number of Unconfident Data inputs to plot
save = whether to save final plots