import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import math
from model import model, layer

def createExpectedOutput(rawOutput: list):
	ret = []
	for x in rawOutput:
		if (x == 'M'):
			ret.append([1, 0])
		else:
			ret.append([0, 1])
	return ret

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
	validRows = np.array(data.iloc[401:])
	validInput = np.array(validRows[:, 2:], dtype=np.float64)
	validInput = (validInput - validInput.mean(axis=0)) / validInput.std(axis=0)
	validOutput = np.array(createExpectedOutput(validRows[:, 1:2]))

	mlp.fillInputsLayer(inputs).fillExpectedOutput(expectedOutput)

	mlp.trainLoop(validInput, validOutput, 100)

	plt.plot(np.array(mlp.costPerEpoch()))
	plt.xlabel("epoch")
	plt.ylabel("trainCost")
	plt.show()

	return

if __name__ == "__main__":
	main()