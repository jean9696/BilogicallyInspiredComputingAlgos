from random import shuffle, random, randint
from time import clock

import math
import matplotlib.pyplot as plt
import matplotlib.lines as lines


# read file
def readFile(filename, separator):
    file = open(filename, "r")
    dataSet = []
    cpt = 0
    min = [99999999999999999999999, 99999999999999999999999]
    max = [0, 0]
    for i in file:
        try:
            if separator is None:
                line = i.split()
            else:
                line = i.split(separator)

            # convert string to float
            for j in range(0, len(line)):
                line[j] = float(line[j]) / 1000
            line.pop(0)

            for j in range(0, len(line)):
                max[j] = line[j] if max[j] < line[j] else max[j]
                min[j] = line[j] if min[j] > line[j] else min[j]

            dataSet.append(line)
            cpt += 1
        except:
            pass
    file.close()
    print(cpt, "read lines in the file", filename)
    return (dataSet, [max[0] + 1, max[1] + 1], [min[0] - 1, min[1] - 1])


# print result with min cost and mean cost
def printResults(results):
    fig, ax1 = plt.subplots()
    ax1.plot(results[0], "-k")
    txt = str(results[0][len(results[0]) - 1])
    ax1.text(len(results[0]), results[0][len(results[0]) - 1], txt)
    ax1.plot(results[1], "-r")
    txt = str(results[1][len(results[1]) - 1])
    ax1.text(len(results[1]), results[1][len(results[1]) - 1], txt)
    plt.show()
    plt.pause(99999999999)


# print map with shorter found path between points to visit everything
def printMap(path, dataSet, gen, max, min):
    fig = plt.figure(2)
    plt.clf()
    map = []
    for i in range(0, len(path) - 1):
        p1 = [(dataSet[path[i]][0] - min[0]) / (max[0] - min[0]),
              (dataSet[path[i + 1]][0] - min[0]) / (max[0] - min[0])]
        p2 = [(dataSet[path[i]][1] - min[1]) / (max[1] - min[1]),
              (dataSet[path[i + 1]][1] - min[1]) / (max[1] - min[1])]
        map.append(lines.Line2D(p2, p1, transform=fig.transFigure, figure=fig))
    fig.lines.extend(map)
    plt.title('Generation number ' + str(gen))
    plt.ion()
    plt.pause(0.0001)


def tspCost(index1, index2, dataSet):
    lat1 = math.pi * dataSet[index1][0] / 180
    lat2 = math.pi * dataSet[index2][0] / 180
    lng1 = math.pi * dataSet[index1][1] / 180
    lng2 = math.pi * dataSet[index2][1] / 180
    q1 = math.cos(lat2) * math.sin(lng1 - lng2)
    q3 = math.sin((lng1 - lng2) / 2)
    q4 = math.cos((lng1 - lng2) / 2)
    q2 = math.sin(lat1 + lat2) * q3 * q3 - math.sin(lat1 - lat2) * q4 * q4
    q5 = math.cos(lat1 - lat2) * q4 * q4 - math.cos(lat1 + lat2) * q3 * q3
    return int(round(6378388 * math.atan2(math.sqrt(q1 * q1 + q2 * q2), q5) + 1))


# store in a matrix all data of path cost so we don't have to calculate them every time
def memoizeTspCost(length):
    memoizeCostMatrix = []
    for i in range(0, length):
        memoizeCostMatrix.append([])
        for j in range(0, length):
            memoizeCostMatrix[i].append(-1)

    def getCost(index1, index2, dataSet):
        if memoizeCostMatrix[index1][index2] > -1:
            return memoizeCostMatrix[index1][index2]
        else:
            cost = tspCost(index1, index2, dataSet)
            memoizeCostMatrix[index1][index2] = cost
            memoizeCostMatrix[index2][index1] = cost
            return cost

    return getCost


# calculate the distance we have for a given path
def calcTotalCost(dataSet, path, getCost):
    costDelta = 0
    for i in range(len(path) - 1):
        costDelta += getCost(path[i], path[i + 1], dataSet)
    return costDelta


# give the list of total distance of given paths
def listOfCosts(dataSet, paths, getCost):
    listRes = []
    for i in range(len(paths)):
        listRes.append(calcTotalCost(dataSet, paths[i], getCost))
    return listRes


# randomly create paths between all points
def createPopulation(length, nPop):
    population = []
    print("Creating population")
    for i in range(0, nPop):
        path = list(range(1, length))
        shuffle(path)  # random population
        population.append([0] + path + [0])
    return population


# Mutate chromosome for a given probability
# Switch with the next chromosome we have to mutate so we change in circle
def mutate(path, pMutate):
    toBeMutated = []
    for i in range(1, len(path) - 1):
        if random() < pMutate:
            toBeMutated.append(i)
    if len(toBeMutated) >= 2:
        tmp = path[toBeMutated[0]]
        for i in range(0, len(toBeMutated) - 1):
            path[toBeMutated[i]] = path[toBeMutated[i + 1]]
        path[toBeMutated[len(toBeMutated) - 1]] = tmp


# Crossover
# Cut parent1 at a random index and add the rest from the parent2
def crossOver(path1, path2):
    iCrossover = int(random() * (len(path1) - 1))
    iChild = iCrossover
    iP2 = 1
    child = path1[0:iCrossover]
    while iChild < len(path1) - 1:
        if path2[iChild] not in child:
            child.append(path2[iChild])
        else:
            while path2[iP2] in child:
                iP2 += 1
            child.append(path2[iP2])
        iChild += 1
    child.append(0)
    return child


def getIndexMaxCost(listInt):
    iMax = 0
    for i in range(1, len(listInt)):
        if listInt[i] >= listInt[iMax]:
            iMax = i
    return iMax


def getIndexMinCost(listInt):
    iMin = 0
    for i in range(1, len(listInt)):
        if listInt[i] <= listInt[iMin]:
            iMin = i
    return iMin


def getMeanCost(costList):
    meanCost = 0
    for i in range(len(costList)):
        meanCost += costList[i]
    return meanCost / len(costList)


def genetic(filename, separator, nPop, pMutate, deltaStop, selectWithRandomParam=False):

    # initialization
    (dataSet, max, min) = readFile(filename, separator)
    parentPop = createPopulation(len(dataSet), nPop)
    cptGen = 0
    cMin = 0
    cMean = deltaStop + 1
    getCost = memoizeTspCost(len(dataSet))
    results = [[], []]

    start = clock()

    print("Starts evolving")
    if selectWithRandomParam:
        print('Select with random param')
    print("Gen.\tMin\ttMean")
    print("----\t---\t----")

    # Continue evolving while the difference between mean and min is under the given delta stop
    while abs(cMin - cMean) / cMean * 100 > deltaStop:
        cptGen += 1

        # create children with random parents and then mutate it
        childPop = []
        for i in range(0, nPop):
            p1 = randint(0, nPop - 1)
            p2 = randint(0, nPop - 1)
            while p1 == p2:
                p1 = randint(0, nPop - 1)
                p2 = randint(0, nPop - 1)
            child = crossOver(parentPop[p1], parentPop[p2])
            mutate(child, pMutate)
            childPop.append(child)

        # Group parents and children
        parentPop.extend(childPop)

        # Create the cost list so we don't have to recalculate after
        listCost = listOfCosts(dataSet, parentPop, getCost)

        if selectWithRandomParam:
            # select 2, keep the best and throw the other
            newPop = []
            while len(
                    newPop) < nPop:
                if len(parentPop) == 1:
                    newPop.append(parentPop[0])
                    parentPop.pop(0)
                else:
                    (i, j) = (0, 0)
                    while i == j or j < i:
                        (i, j) = (randint(0, len(parentPop) - 1), randint(0, len(parentPop) - 1))
                    newPop.append(
                        parentPop[i] if listCost[i] < listCost[j] else
                        parentPop[j])
                    parentPop.pop(i)
                    listCost.pop(i)
                    listCost.pop(j - 1)
                    parentPop.pop(j - 1)
            parentPop = newPop
        else:
            # keep only bests
            while len(parentPop) > nPop:
                iMax = getIndexMaxCost(listCost)
                parentPop.pop(iMax)
                listCost.pop(iMax)

        # Calculate cost of new population
        listCost = listOfCosts(dataSet, parentPop, getCost)

        if cptGen % 100 == 1:
            printMap(parentPop[getIndexMinCost(listCost)], dataSet, cptGen, max, min)

        cMin = listCost[getIndexMinCost(listCost)]
        cMean = getMeanCost(listCost)
        print(cptGen, cMin, round(cMean, 5))

        results[0].append(cMin)
        results[1].append(cMean)

    stop = clock()
    print("Execution time :", stop - start)
    printMap(parentPop[getIndexMinCost(listCost)], dataSet, cptGen, max, min)
    printResults(results)


genetic('cities', ' ', 500, 0.2, 10, False)
