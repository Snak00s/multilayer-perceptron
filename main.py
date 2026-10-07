import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model import model, layer

	# def __backPropagation(self):
	# 	N = self._sampleAmount
	# 	L = len(self._layers)
	# 	grads = [None] * (L - 1)

	# 	delta = self._layers[-1].activate() - self._expectedOutput
	# 	for l in range(L - 1, 0, -1):
	# 		grads[l - 1] = delta.T @ self._layers[l - 1].activate() / N
	# 		if l > 1:
	# 			a = self._layers[l - 1].activate()
	# 			delta = (delta @ self._layers[l].prevMatrix()) * a * (1 - a)

	# 	for l in range(1, L):
	# 		self._layers[l]._prevMatrix -= self._learningRate * grads[l - 1]
	# 	return self

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
		data = data.sample(frac=1)
		dataRows = np.array(data.iloc[0:400])
		inputs = np.array(dataRows[:, 2:], dtype=np.float64)
		inputs = (inputs - inputs.mean(axis=0)) / inputs.std(axis=0)

		expectedOutput = np.array(createExpectedOutput(dataRows[:, 1:2]))
		mlp = model.createNetwork([
			layer.createLayer(idx=0, size=30),
			layer.createLayer(idx=1, size=24),
			layer.createLayer(idx=2, size=24),
			layer.createLayer(idx=3, size=2)
		]).fillTrainIO(inputs, expectedOutput)

		validRows = np.array(data.iloc[401:])
		validInput = np.array(validRows[:, 2:], dtype=np.float64)
		validInput = (validInput - validInput.mean(axis=0)) / validInput.std(axis=0)
		validOutput = np.array(createExpectedOutput(validRows[:, 1:2]))

		mlp.trainLoop(validInput, validOutput, 1000)

		# print("----------------------------Cost----------------------------")
		# print(f"Begin cost :		train = {mlp.costPerEpoch()[0]:.3f}, valid = {mlp.validCostPerEpoch()[0]:.3f}")
		# print(f"End cost :		train = {mlp.costPerEpoch()[-1]:.3f}, valid = {mlp.validCostPerEpoch()[-1]:.3f}")
		# print("--------------------------Accuracy--------------------------")
		# print(f"Begin accuracy :	train = {mlp.accuracy()[0]:.3f}, valid = {mlp.validAccuracy()[0]:.3f}")
		# print(f"End accuracy :		train = {mlp.accuracy()[-1]:.3f}, valid = {mlp.validAccuracy()[-1]:.3f}")
		# print("------------------------------------------------------------")


		plt.figure()
		plt.plot(np.array(mlp.costPerEpoch()))
		plt.plot(np.array(mlp.validCostPerEpoch()))
		plt.xlabel("epoch")
		plt.ylabel("Cost")
		plt.legend(['Train cost', 'Validation cost'])
		# plt.legend(['Train cost'])

		plt.figure()
		plt.plot(np.array(mlp.accuracy()))
		plt.plot(np.array(mlp.validAccuracy()))
		plt.title("Accuracy")
		plt.xlabel("epoch")
		plt.ylabel("Accuracy")
		plt.legend(['Train accuracy', 'Validation accuracy'])
		# plt.legend(['Train accuracy'])

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