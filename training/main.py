import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

from training.model import model, layer
from utils.parser import Parser

def help():
	print("model trainer, this program will train on the given data generated from 'data_slicer.data_slicer'.")
	print("\t--data:\t\t\t'file.csv' set the training data (THOSE DATA SHOULD BE PREPROSSESED BY 'data_slicer').")
	print("\noptionnal:")
	print("\t--layer:\t\t'int' 'int' 'int', determine the shape of the model.")
	print("\t--epochs:\t\tset the ammount of time the model will train on the given data.")
	print("\t--loss:\t\t\tset the loss function to use (there is only 'categoricalCrossentropy' lol).")
	print("\t--learning_rate:\tset the learning rate (usualy 0 < lr < 1).")
	print("\t--batch_size:\t\tset the batch size.")
	return

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
		#------------------------Parsing----------------------

		parser = Parser()

		parser.special_help(help)
		parser.add_arg("data", ammount=1, argType=str)
		parser.add_arg("layer", ammount=0, argType=int, optionnal=True)
		parser.add_arg("epochs", ammount=1, argType=int, optionnal=True)
		parser.add_arg("loss", ammount=1, argType=str, optionnal=True)
		parser.add_arg("learning_rate", ammount=1, argType=float, optionnal=True)
		parser.add_arg("batch_size", ammount=1, argType=int, optionnal=True)

		args = parser.parse_args()

		#---------------------Data Setting--------------------

		data = pd.read_csv(args['data'][0], header=None)
		assert (data.iloc[:, -1].dtype == 'str'
				and data.iloc[:, -1].str.fullmatch("train|valid", case=True).all()),	"Invalid dataset, please use dataset created by data_slicer.data_slicer." 
		idx_colonne = data.select_dtypes(include=['str']).apply(
			lambda col: col.str.fullmatch("B|M", case=True).all()
		)
		idx_output = len(idx_colonne.loc[idx_colonne.iloc[:] == True].index)
		assert idx_output == 1,															"Invalid dataset, missing the expected output columns."
		assert len(data.columns) == 33,													"Invalid dataset, it should have 33 columns, please use dataset created by data_slicer.data_slicer."

		siz = len(data.columns)
		cancerType = data.pop(idx_output)
		dataType = data.pop(siz - 1)
		data = (data - data.mean(axis=0, numeric_only=True)) / data.std(axis=0, numeric_only=True)
		data.insert(siz - 2, f"{siz - 2}", dataType)

		trainmask = data.iloc[:, -1] == 'train'
		validmask = data.iloc[:, -1] == 'valid'

		train = data.loc[trainmask].iloc[:, 1:-1]
		valid = data.loc[validmask].iloc[:, 1:-1]

		cancerTrain = cancerType.loc[trainmask]
		cancervalid = cancerType.loc[validmask]

		trainInputs = np.array(train, dtype=np.float64)
		trainOuputs = np.array(createExpectedOutput(np.array(cancerTrain)))

		validInputs = np.array(valid, dtype=np.float64)
		validOuputs = np.array(createExpectedOutput(np.array(cancervalid)))

		print("x_train shape :", trainInputs.shape)
		print("x_valid shape :", validInputs.shape)

		#-----------------------Creation----------------------

		laySize = args['layer']
		if (len(laySize) == 0):
			laySize = [len(trainInputs[0]), 24, 24, len(trainOuputs[0])]
		else:
			laySize.append(len(trainOuputs[0]))
			laySize.insert(0, len(trainInputs[0]))

		for siz in laySize:
			assert siz > 0, "All layers should have 1 or more neuron."

		mlp = model.createNetwork([
			layer.createLayer(size=laySize[i], activation='sigmoid')
			if i < len(laySize) - 1
			else
			layer.createLayer(size=laySize[i], activation='softmax')
			for i in range(len(laySize))
			])

		#---------------------Param init----------------------
		lr = args['learning_rate']
		if (len(lr) == 0):
			lr = 0.0314
		else:
			lr = args['learning_rate'][0]

		loss = args['loss']
		if (len(loss) == 0):
			loss = 'categoricalCrossentropy'
		else:
			loss = args['loss'][0]

		batchSize = args['batch_size']
		if (len(args['batch_size']) == 0):
			batchSize = 8
		else:
			batchSize = args['batch_size'][0]

		epochs = args['epochs']
		if (len(args['epochs']) == 0):
			epochs = 84
		else:
			epochs = args['epochs'][0]

		#-----------------------Training----------------------
		mlp.fit(
				trainData=(trainInputs, trainOuputs),
				validData=(validInputs, validOuputs),
				learningRate=lr,
				lossFunc=loss,
				batchSize=batchSize,
				epoch=epochs,
				verbose=True
			)

		#-----------------------Plotting----------------------

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