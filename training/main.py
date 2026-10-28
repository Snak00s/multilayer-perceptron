import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

from training.model import model, layer
from utils.parser import Parser

def help():
	print("model trainer, this program will train on the given data generated from 'data_separation.data_repartition'.")
	print("\t--data:\t\t\t'mlp_data.npy' set the training data.")
	print("\t--layer:\t\tint int int, determine the shape of the model.")
	print("\t--epochs:\t\tset the ammount of time the model will train on the given data.")
	print("\t--loss:\t\t\tset the loss function to use (there is only 'categoricalCrossentropy' lol).")
	print("\t--learning_rate:\tset the learning rate (usualy 0 < lr < 1).")
	print("\t--batch_size:\t\tset the batch size.")
	return

def main():
	try:
		# -----------------------Parsing----------------------

		parser = Parser()

		parser.special_help(help)
		parser.add_arg("layer", ammount=0, argType=int)
		parser.add_arg("epochs", ammount=1, argType=int)
		parser.add_arg("loss", ammount=1, argType=str)
		parser.add_arg("learning_rate", ammount=1, argType=float)
		parser.add_arg("batch_size", ammount=1, argType=int)

		args = parser.parse_args()

		#-----------------------Training----------------------

		data = np.load('data/mlp_data.npy', allow_pickle=True)

		data = data.item()

		trainData : tuple = data['train']
		validData : tuple = data['valid']

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
				lossFunc=args['loss'][0],
				batchSize=args['batch_size'][0],
				epoch=args['epochs'][0],
				verbose=True
			)

		#-----------------------------------------------------

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

		path = Path("data/mlp.npy")
		path.parent.mkdir(parents=True, exist_ok=True)
		np.save(path, mlp)

	except Exception as AE:
		print(f"{type(AE).__name__}: {AE}", file=sys.stderr)
		exit(1)

	except KeyboardInterrupt:
		print("\nKeyboard Interuption", file=sys.stderr)

	return

if __name__ == "__main__":
	main()