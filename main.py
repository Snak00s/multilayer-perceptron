import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import math
from model import model, layer

def sigNumber(x, n):
	if x == 0:
		return 0
	return round(x, n - 1 - int(math.floor(math.log10(abs(x)))))

def weightDot(input_lst: list, weight_lst: list):
	"""Calculate the node value before the activation func"""
	return np.dot(input_lst, weight_lst) + 1

def catCrossEntropy(expectedOutput: list, outLayerOutput: list):
	return -(np.sum([expectedOutput[i] * np.log(outLayerOutput[i]) for i in range(len(expectedOutput))]))

def createExpectedOutput(rawOutput: list):
	ret = []
	for x in rawOutput:
		if (x == 'M'):
			ret.append([1, 0])
		else:
			ret.append([0, 1])
	return ret

def epochCost(res: np.array, expRes: np.array):
	losses = -((expRes * np.log(res)) + (1 - expRes) * np.log(1 - res))
	m = len(res)
	cost = (1 / m) * losses
	return np.sum(cost)

def main():

	data = pd.read_csv("data.csv", header=None).reset_index(drop="True")
	dataRows = np.array(data.iloc[0:400])
	inputs = np.array(dataRows[:, 2:], dtype=np.float64)
	inputs = (inputs - inputs.mean(axis=0)) / inputs.std(axis=0)

	expectedOutput = np.array(createExpectedOutput(dataRows[:, 1:2]))
	mlp = model.createNetwork([
		layer.createLayer(idx=0, size=30),
		layer.createLayer(idx=1, size=24),
		layer.createLayer(idx=2, size=24),
		layer.createLayer(idx=3, size=2)
	])

	mlp.fillInputsLayer(inputs).fillExpectedOutput(expectedOutput)

	mlp.trainLoop(100)

	# print(mlp._layers[-1].activate())

	# print(epochCost(mlp._layers[-1].activate(), expectedOutput))

	# print(mlp.epochCost())

	# mlp.backPropagation()

	return

if __name__ == "__main__":
	main()