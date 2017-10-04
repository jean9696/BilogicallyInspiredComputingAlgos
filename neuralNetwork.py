import numpy as np
import time
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

nIn = 1
nHidden = 5
nOut = 1

nSamples = 5000

learningRate = 0.01
momentum = 0.9


# activation function
def sigmoid(x, deriv=False):
    if deriv:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


# input data, transpose, layer 1, layer 2, biases
def train(x, t, v, w, bV, bW):
    # forward -- matrix multiply + biases
    A = np.dot(x, v) + bV
    Z = sigmoid(A)

    B = np.dot(Z, w) + bW
    Y = sigmoid(B)

    # backward
    Ew = Y - t
    Ev = sigmoid(A, True) * np.dot(w, Ew)

    # predict our cost
    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)

    # mean squared error
    cost = ((Y - t) ** 2).mean()

    return cost, (dV, dW, Ev, Ew)


# input data, transpose, layer 1, layer 2, biases
def predict(x, v, w, bV, bW):
    A = np.dot(x, v) + bV
    B = np.dot(sigmoid(A), w) + bW
    return sigmoid(B)


# create layers
V = np.random.normal(scale=0.1, size=(nIn, nHidden))
W = np.random.normal(scale=0.1, size=(nHidden, nOut))

bV = np.zeros(nHidden)
bW = np.zeros(nOut)

params = [V, W, bV, bW]


def functionToLearn(x):
    return np.cos(x)


def normalize(x, back=False):
    if back:
        return x * 10.0 - 5.0
    else:
        return np.true_divide(x + 5, 10)


def normalizeData(data, back=False):
    x = []
    for y in data:
        x.append(normalize(y, back))
    return x


# generate data
dataIn = np.random.uniform(-5, 5, size=(nSamples, nOut))
dataOut = functionToLearn(dataIn)
normalizedDataIn = normalizeData(dataIn)
normalizedDataOut = normalizeData(dataOut)


# compare
def compare(params, title):
    plt.clf()
    predictions = []
    realResults = []
    index = []
    for y in np.random.uniform(-5, 5, size=(100, nOut)):
        predictions.append(predict(normalize(y[0]), *params))
        index.append(y[0])
        realResults.append(functionToLearn(y[0]))

    fig = plt.figure(1)

    ax1 = fig.add_subplot(211)
    ax1.scatter(index, realResults, c='r')
    ax1.scatter(index, normalizeData(predictions, True), c='b')
    blue_patch = mpatches.Patch(color='blue', label='Predicted results')
    red_patch = mpatches.Patch(color='red', label='Real results')
    plt.legend(handles=[blue_patch, red_patch])
    ax1.grid(True)
    plt.title(title)

    plt.ion()
    plt.pause(0.0001)


# training
for c in range(800):
    err = []
    upd = [0] * len(params)

    t0 = time.clock()
    # for each data point, update our weights
    for i in range(dataIn.shape[0]):
        cost, grad = train(normalizedDataIn[i], normalizedDataOut[i], *params)
        # update cost
        for j in range(len(params)):
            params[j] -= upd[j]
            upd[j] = learningRate * grad[j] + momentum * upd[j]

        err.append(cost)
    if c % 10 == 0:
        compare(params, 'Training loop number %d, Cost: %.8f' % (c + 1, np.mean(err)))
    print('Epoch:%d, Cost: %.8f, Time: %fs' % (c + 1, np.mean(err), time.clock() - t0))

plt.show()
plt.pause(99999999999)

