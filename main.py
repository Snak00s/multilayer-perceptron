import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

from model import model, layer
from parser import Parser

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
		# -----------------------Parsing----------------------

		parser = Parser()

		parser.add_arg("layer", ammount=0, argType=int)
		parser.add_arg("epochs", ammount=1, argType=int)
		parser.add_arg("loss", ammount=1, argType=str)
		parser.add_arg("learning_rate", ammount=1, argType=float)
		parser.add_arg("batch_size", ammount=1, argType=int)

		args = parser.parse_args()

		# ------------------Data Repartition------------------

		data = pd.read_csv("data.csv", header=None).reset_index(drop="True")
		data = data.sample(frac=1)
		dataRows = np.array(data.iloc[0:400])
		inputs = np.array(dataRows[:, 2:], dtype=np.float64)
		inputs = (inputs - inputs.mean(axis=0)) / inputs.std(axis=0)
		expectedOutput = np.array(createExpectedOutput(dataRows[:, 1:2]))

		validRows = np.array(data.iloc[401:])
		validInput = np.array(validRows[:, 2:], dtype=np.float64)
		validInput = (validInput - validInput.mean(axis=0)) / validInput.std(axis=0)
		validOutput = np.array(createExpectedOutput(validRows[:, 1:2]))

		#-----------------------Training----------------------

		laySize = args['layer']
		laySize.append(len(expectedOutput[0]))
		laySize.insert(0, len(inputs[0]))

		mlp = model.createNetwork([
			layer.createLayer(size=laySize[i], activation='sigmoid')
			if i < len(laySize) - 1
			else
			layer.createLayer(size=laySize[i], activation='softmax')
			for i in range(len(laySize))
			])

		mlp.fit(
			trainData=(inputs, expectedOutput),
			validData=(validInput, validOutput),
			learningRate=args['learning_rate'][0],
			batchSize=len(inputs),
			epoch=args['epochs'][0],
			verbose=True
			)

		#-----------------------Print-------------------------

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
		plt.title("Cost")
		plt.xlabel("epoch")
		plt.ylabel("Cost")
		plt.legend(['Train cost', 'Validation cost'])

		plt.figure()
		plt.plot(np.array(mlp.accuracy()))
		plt.plot(np.array(mlp.validAccuracy()))
		plt.title("Accuracy")
		plt.xlabel("epoch")
		plt.ylabel("Accuracy")
		plt.legend(['Train accuracy', 'Validation accuracy'])
		# np.save("mlp", mlp)

		# test = np.load("mlp.npy", allow_pickle=True)
		# test = test.item()
		# print(test._sampleAmount)

		plt.show()

		#-----------------------------------------------------

	except Exception as AE:
		print(f"{type(AE).__name__}: {AE}", file=sys.stderr)
		exit(1)

	except KeyboardInterrupt:
		print("\nKeyboard Interuption", file=sys.stderr)

	return

if __name__ == "__main__":
	main()