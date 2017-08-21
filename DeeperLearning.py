'''
Created on Jul 10, 2017

@author: jackgellerman
'''
import numpy as np
import sys


def nonLinearFunction(x, functionType):
    output = 0
    if functionType == 3:#sigmoid
        output = 1/(1+np.exp(-x))
    if functionType == 2:#ReLU
        output = np.max(0, x)
    if functionType == 1:
        output = np.tanh(x)
    return output

def nonLinearFunctionDerivative(x, functionType):
    output = 0
    if functionType == 3:
        output = nonLinearFunction(x, functionType)*(1-(nonLinearFunction(x, functionType)))
    if functionType == 2:
        if x>0:
            output = 1
        output = np.max(0, x)
    if functionType == 1:
        output = np.cosh(x)
    return output
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])#4x3
y = np.array([[0],[1],[1],[0]])
# random initialization of weights
np.random.seed(1)
hiddenLayerSize = 4
#weightsInputToHidden = 2*np.random.random((3,hiddenLayerSize)) - 1
allWeights = []
hiddenLayersNumber = 1
allWeights.append((2*np.random.random((3,hiddenLayerSize)) - 1))
for i in range(1,hiddenLayersNumber):    
    allWeights.append( (2*np.random.random((hiddenLayerSize,hiddenLayerSize)) - 1))
allWeights.append( (2*np.random.random((hiddenLayerSize,1)) -1))
hiddenLayers = [None] * (hiddenLayersNumber)
deltas = [None] * (hiddenLayersNumber+1)
allErrors = [None] * (hiddenLayersNumber+1)
for i in xrange(60000):#epochs
    #print(i)
    layerZero = X #column refers to a node, so three nodes in input layer (to be used for every example sample)
    #compute the dot product,  forward pass
    hiddenLayerOne = nonLinearFunction(np.dot(layerZero,allWeights[0]),3)
    hiddenLayers[0] = hiddenLayerOne
    for z in range(1,hiddenLayersNumber): 
        #print(i)   
        hiddenLayers[z] = nonLinearFunction(np.dot(hiddenLayers[z-1],allWeights[z]),3)
    #print(len(hiddenLayers) -1)
    outputLayer = nonLinearFunction(np.dot(hiddenLayers[len(hiddenLayers) -1],allWeights[len(allWeights)-1]),3)
    overallError = outputLayer - y
    ##move this lower
    for g in range(0,hiddenLayersNumber+1):
        if g == 0:
            #print(i)
            if (i%10000) == 0:
                print "Error after "+str(i)+" epochs: "+str(np.mean(np.abs(overallError)))
                print(sys.argv[0])
            someDelta = overallError*nonLinearFunctionDerivative(np.dot(hiddenLayers[len(hiddenLayers)-1],allWeights[len(allWeights)-1]),3)
            allErrors[hiddenLayersNumber] = overallError
            deltas[hiddenLayersNumber] = someDelta
        if g > 0:
            if g ==  hiddenLayersNumber:
                #print(len(allWeights))
                #print("weights")
           
                firstLayerError = deltas[len(deltas)-g].dot(allWeights[len(allWeights)-g].T)
            #print(((allWeights)))
                firstLayerDelta = firstLayerError*nonLinearFunctionDerivative(np.dot(layerZero,allWeights[len(allWeights)-g-1]),3)
                allErrors[hiddenLayersNumber-g] = firstLayerError
                deltas[hiddenLayersNumber-g] = firstLayerDelta
            else:
                #print("lloop")
                backPropError = deltas[len(deltas)-g].dot(allWeights[len(allWeights)-g].T)
                backPropDelta = backPropError*nonLinearFunctionDerivative(np.dot(hiddenLayers[len(hiddenLayers)-g-1],allWeights[len(allWeights)-g-1]),3)
                allErrors[hiddenLayersNumber-g] = backPropError
                deltas[hiddenLayersNumber-g] = backPropDelta
    for c in range(len(allWeights)):
        if c == 0:
            allWeights[c] -= layerZero.T.dot(deltas[c])
        else:
            allWeights[c] -= hiddenLayers[c-1].T.dot(deltas[c])
print "Output layer After Training:"
print outputLayer
##classify some other example

