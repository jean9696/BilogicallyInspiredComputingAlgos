import numpy as np
import time
from matplotlib import pyplot as plt

nIn = 1
nHidden = 5
nOut = 1

nSamples = 100

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
    # print(x, t, Y)
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


# generate data
dataIn = np.random.rand(nSamples, nIn)
dataOut = functionToLearn(dataIn)
print(dataIn, dataOut)

# training
for epoch in range(100):
    err = []
    upd = [0] * len(params)

    t0 = time.clock()
    # for each data point, update our weights
    for i in range(dataIn.shape[0]):
        cost, grad = train(dataIn[i], dataOut[i], *params)
        # update cost
        for j in range(len(params)):
            params[j] -= upd[j]

        for j in range(len(params)):
            upd[j] = learningRate * grad[j] + momentum * upd[j]

        err.append(cost)

    print('Epoch:%d, Cost: %.8f, Time: %fs' % (epoch + 1, np.mean(err), time.clock() - t0))

# compare
predictions = []
realResults = []
index = []
for y in dataIn:
    predictions.append(predict(y[0], *params))
    index.append(y[0])
    realResults.append(functionToLearn(y[0]))

fig = plt.figure(1)

ax1 = fig.add_subplot(211)
ax1.scatter(index, realResults, c='r')
ax1.scatter(index, predictions, c='b')
ax1.grid(True)

plt.show()
