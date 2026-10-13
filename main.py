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

		#-----------------------Training----------------------

		data = np.load('mlp_data.npy', allow_pickle=True)

		data = data.item()

		trainData : tuple = data['train']
		validData : tuple = data['valid']
		# testData : tuple = data['test']

		laySize = args['layer']
		laySize.append(len(trainData[1][0]))
		laySize.insert(0, len(trainData[0][0]))

		mlp = model.createNetwork([
			layer.createLayer(size=laySize[i], activation='sigmoid')
			if i < len(laySize) - 1
			else
			layer.createLayer(size=laySize[i], activation='softmax')
			for i in range(len(laySize))
			])

		mlp.fit(
				trainData=trainData,
				validData=validData,
				learningRate=args['learning_rate'][0],
				batchSize=args['batch_size'][0],
				epoch=args['epochs'][0],
				verbose=True
			)

		#-----------------------------------------------------

		# mlp.useModel(testData=testData)

		plt.figure()
		plt.plot(np.array(mlp.costPerEpoch()))
		plt.plot(np.array(mlp.validCostPerEpoch()))
		plt.title("Cost")
		plt.xlabel("epoch")
		plt.ylabel("Cost")
		plt.legend(['Train cost', 'Validation cost'])
		plt.savefig("cost.png")

		plt.figure()
		plt.plot(np.array(mlp.accuracy()))
		plt.plot(np.array(mlp.validAccuracy()))
		plt.title("Accuracy")
		plt.xlabel("epoch")
		plt.ylabel("Accuracy")
		plt.legend(['Train accuracy', 'Validation accuracy'])
		plt.savefig("accuracy.png")

		#-----------------------------------------------------
		np.save("mlp", mlp.exportModel())

	except Exception as AE:
		print(f"{type(AE).__name__}: {AE}", file=sys.stderr)
		exit(1)

	except KeyboardInterrupt:
		print("\nKeyboard Interuption", file=sys.stderr)

	return

if __name__ == "__main__":
	main()