import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model import model, layer

def createExpectedOutput(rawOutput: list):
	ret = []
	for x in rawOutput:
		if (x == 'M'):
			ret.append([1, 0])
		elif (x == 'B'):
			ret.append([0, 1])
		else:
			raise AssertionError(f"Invalid output : {x}, should be 'B' or 'M'")
	return ret

def main():
	try:
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

		mlp.trainLoop(validInput, validOutput, 1000)

		plt.figure()
		plt.plot(np.array(mlp.costPerEpoch()))
		plt.title("Cost")
		plt.xlabel("epoch")
		plt.ylabel("trainCost")

		plt.figure()
		plt.plot(np.array(mlp.accuracy()))
		plt.title("Accuracy")
		plt.xlabel("epoch")
		plt.ylabel("accuracy")

		plt.show()
	except AssertionError:
		print("Assert error")
		exit(1)
	except KeyboardInterrupt:
		print("KeyboardInterrupt error")
		exit(1)

	return

if __name__ == "__main__":
	main()