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

def createInterLayerMatrix(prevLayerSize: int, actualLayerSize: int) -> list :
	upLim = np.sqrt(6 / actualLayerSize)
	downLim = - np.sqrt(6 / prevLayerSize)
	return [[sigNumber(float(random.uniform(downLim, upLim)), 2) for _ in range(prevLayerSize)] for _ in range(actualLayerSize)]

def softMax(lst: list) -> list:
	s = np.sum([np.exp(x) for x in lst])
	return [np.exp(x) / s for x in lst]

def catCrossEntropy(expectedOutput: list, outLayerOutput: list):
	return -(np.sum([expectedOutput[i] * np.log(outLayerOutput[i]) for i in range(len(expectedOutput))]))

def main():
	data = pd.read_csv("data.csv", header=None).sort_values(0).reset_index(drop="True")
	inputs = data.iloc[0].to_list()[2:]
	mlp = model.createNetwork([
		layer.createLayer(idx=0, size=30),
		layer.createLayer(idx=1, size=24),
		layer.createLayer(idx=2, size=24),
		layer.createLayer(idx=3, size=2)
	])

	mlp.fillInputsLayer(inputs).fillExpectedOutput([1, 0]).forwardPropagation()

	test = softMax(mlp.layers()[3].nodes)
	#we suppose we wanted [1, 0]
	print(len(mlp.layers()[1].prevMatrix()[0]))
	# print(catCrossEntropy([1, 0], test))

	return

if __name__ == "__main__":
	main()