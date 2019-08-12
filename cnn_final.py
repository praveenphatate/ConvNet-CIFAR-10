''' CNN Final '''
import numpy as np
from layers import Conv, ReLU, Maxpool, Flatten, FullyConnected
from Cnn import CNN
from sklearn.utils import shuffle
import os
import _pickle as pickle
import matplotlib.pyplot as plt

lossPlot = []
numIterPlot = []
trainPlot = []
testPlot = []

def get_minibatches(X, y, minibatch_size,shuffles=True):
    m = X.shape[0]
    minibatches = []
    if shuffles:
        X, y = shuffle(X, y)
    for i in range(0, m, minibatch_size):
        X_batch = X[i:i + minibatch_size, :, :, :]
        y_batch = y[i:i + minibatch_size, ]
        minibatches.append((X_batch, y_batch))
    return minibatches

def vanilla(params, grads, learning_rate=0.01):
    for param, grad in zip(params, reversed(grads)):
        for i in range(len(grad)):
            param[i] += - learning_rate * grad[i]


def stochastic_gradient(cnn, X_train, y_train, minibatch_size, epoch, learning_rate, verbose=True,
        X_test=None, y_test=None):
    minibatches = get_minibatches(X_train, y_train, minibatch_size)
    for i in range(epoch):
        loss = 0
        if verbose:
            print("Epoch {0}".format(i + 1))
        for X_mini, y_mini in minibatches:
            loss, grads = cnn.train_step(X_mini, y_mini)
            vanilla(cnn.params, grads, learning_rate=learning_rate)
        
        if verbose:
            train_acc = accuracy(y_train, cnn.predict(X_train))
            test_acc = accuracy(y_test, cnn.predict(X_test))
            print("Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}".format(loss, train_acc, test_acc))
            trainPlot.append(train_acc)
            testPlot.append(test_acc)
            lossPlot.append(loss)
            numIterPlot.append(i)
    plt.plot(numIterPlot, lossPlot)
    plt.title("Loss vs epoch")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.show()
    plt.plot(numIterPlot, trainPlot)
    plt.title("Training Accuracy vs epoch")
    plt.xlabel("epoch")
    plt.ylabel("Training Accuracy")
    plt.show()
    plt.plot(numIterPlot, testPlot)
    plt.title("Test Accuracy vs epoch")
    plt.xlabel("epoch")
    plt.ylabel("Test Accuracy")
    plt.show()
    return cnn


def make_cnn(X_dim, num_class):
    conv = Conv(X_dim, n_filter=16, h_filter=5,w_filter=5, stride=1, padding=2)
    relu = ReLU()
    maxpool = Maxpool(conv.out_dim, size=2, stride=2)
    conv2 = Conv(maxpool.out_dim, n_filter=20, h_filter=5,w_filter=5, stride=1, padding=2)
    relu2 = ReLU()
    maxpool2 = Maxpool(conv2.out_dim, size=2, stride=2)
    flat = Flatten()
    fc = FullyConnected(np.prod(maxpool2.out_dim), num_class)
    
    return [conv, relu, maxpool, conv2, relu2, maxpool2, flat, fc]

def load_CIFAR_batch(filename):
    ''' load single batch of cifar '''
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding ='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).astype("float")
        Y = np.array(Y)
        return X, Y

def load(ROOT):
    ''' load all of cifar '''
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return (Xtr, Ytr), (Xte, Yte)

def one_hot_encode(y, num_class):
    m = y.shape[0]
    onehot = np.zeros((m, num_class), dtype="int32")
    for i in range(m):
        idx = y[i]
        onehot[i][idx] = 1
    return onehot

def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)  # both are not one hot encoded

if __name__ == "__main__":

    training_set, test_set = load(r'C:\Users\prave\Assignment One\cifar-10-python\cifar-10-python\cifar-10-batches-py')
    X, y = training_set
    X_test, y_test = test_set
    X_train = X[:1000]/255.
    y_train = y[:1000].astype(int)
    X_test = X_test[:100]/255.
    y_test = y_test[:100].astype(int)
    cifar10_dims = (3, 32, 32)
    cnn = CNN(make_cnn(cifar10_dims, num_class=10))
    cnn = stochastic_gradient(cnn, X_train, y_train, minibatch_size=10, epoch=20,
                       learning_rate=0.01, X_test=X_test, y_test=y_test)