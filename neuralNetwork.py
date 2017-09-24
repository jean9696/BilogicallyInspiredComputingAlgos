import numpy as np
from matplotlib import pyplot as plt
from scipy import misc


# get a number between 0 and 1
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# give a number between 0 and 1 from parameters
# todo: change 2 parameters into list
def getPrediction(m1, m2, w1, w2, b):
    return sigmoid(m1 * w1 + m2 * w2 + b)


# cost of the neural network, our goal is to make it close to 0
def getCost(target):
    def getCost2(pred):
        return np.square(pred - target)
    return getCost2


# get the derive of the number we have to subtract to make the cost closer to 0
def slope(b, target):
    return misc.derivative(getCost(target), b)


# training neural network
def reduceCost(b, target, step=0.1):
    return b - step * slope(b, target)


dataSet = [
    [3, 1.5, 1],
    [2, 1, 0],
    [4, 1.5, 1],
    [3, 1, 0],
    [3.5, .5, 1],
    [2, .5, 0],
    [5.5, 1, 1],
    [1, 1, 0]
]

mystery = [4.5, 1]

# show dataSet
for i in range(len(dataSet)):
    point = dataSet[i]
    plt.scatter(point[0], point[1], c="r" if point[2] == 1 else "b")

X = np.linspace(0, 5, 100)
Y = sigmoid(X)
plt.plot(X, Y)
plt.axis([0, 6, 0, 6])
plt.grid()


w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

# training loop
for i in range(1000):
    randomIndex = np.random.randint(len(dataSet))
    randomPoint = dataSet[randomIndex]
    pred = getPrediction(randomPoint[0], randomPoint[1], w1, w2, b)

    target = randomPoint[2]
    cost = getCost(pred)(target)

    costSlope = slope(pred, target)

    print(cost)

plt.show()
