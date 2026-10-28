import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

from bonus.modelBonus import model, layer, confMatrix
from utils.parser import Parser

def help():
	print("model trainer, this program will train on the given data generated from 'data_separation.data_repartition'.")
	print("\t--data:\t\t\t'mlp_data.npy' set the training data.")
	print("\t--layer:\t\tint int int, determine the shape of the model.")
	print("\t--epochs:\t\tset the ammount of time the model will train on the given data.")
	print("\t--loss:\t\t\tset the loss function to use (there is only 'categoricalCrossentropy' lol).")
	print("\t--learning_rate:\tset the learning rate (usualy 0 < lr < 1).")
	print("\t--batch_size:\t\tset the batch size.")
	print("\t--optimizer:\t\tset the training algorith; to use.")
	print("\t--early_stop:\t\t(int, float) with 0 < float < 1, special instruction, will stop the training if the model cost start to converge.")
	return

def main():
	try:
		# -----------------------Parsing----------------------

		parser = Parser()
		parser.special_help(help)
		parser.add_arg("data", ammount=1, argType=str)
		parser.add_arg("layer", ammount=0, argType=int)
		parser.add_arg("epochs", ammount=1, argType=int)
		parser.add_arg("loss", ammount=1, argType=str)
		parser.add_arg("learning_rate", ammount=1, argType=float)
		parser.add_arg("batch_size", ammount=1, argType=int)
		parser.add_arg("optimizer", ammount=1, argType=str, optionnal=True)
		parser.add_arg("early_stop", ammount=1, argType=tuple, optionnal=True)

		args = parser.parse_args()

		#-----------------------Training----------------------

		data = np.load(args['data'][0], allow_pickle=True)

		data = data.item()

		trainData: tuple = data['train']
		validData: tuple = data['valid']

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

		#Optimizer setting (Bonus)
		optimizer = ''
		if (args.get('optimizer')):
			optimizer = args['optimizer'][0]

		#Early Stop setting (Bonus)
		earlyStop = (int(0), float(0))
		if (args.get('early_stop')):
			earlyStop = args['early_stop'][0]
			earlyStop = (int(earlyStop[0]), float(earlyStop[1]))

		mlp.fit(
				trainData=trainData,
				validData=validData,
				learningRate=args['learning_rate'][0],
				lossFunc=args['loss'][0],
				batchSize=args['batch_size'][0],
				epoch=args['epochs'][0],
				verbose=True,
				optimizer=optimizer,
				earlyStopping=earlyStop
			)

		#-----------------------------------------------------

		path = f"data/{mlp.synthetic_name()}/"
		model_path = Path(path + "mlp.npy")
		model_path.parent.mkdir(parents=True, exist_ok=True)
		np.save(model_path, mlp)

		plt.figure()
		plt.plot(np.array(mlp.costPerEpoch()))
		plt.plot(np.array(mlp.validCostPerEpoch()))
		plt.title("Cost")
		plt.xlabel("epoch")
		plt.ylabel("Cost")
		plt.legend(['Train cost', 'Validation cost'])
		plt.savefig(path + "cost.png")

		plt.figure()
		plt.plot(np.array(mlp.accuracy()))
		plt.plot(np.array(mlp.validAccuracy()))
		plt.title("Accuracy")
		plt.xlabel("epoch")
		plt.ylabel("Accuracy")
		plt.legend(['Train accuracy', 'Validation accuracy'])
		plt.savefig(path + "accuracy.png")

		plt.figure()
		plt.plot(np.array(mlp.precision()))
		plt.plot(np.array(mlp.recall()))
		plt.plot(np.array(mlp.f1_score()))
		plt.title("F1_score")
		plt.xlabel("epoch")
		plt.ylabel("value")
		plt.legend(['Precision', 'Recall', 'F1_score'])
		plt.savefig(path + "f1_score.png")

		plt.figure()
		plt.plot(np.array(mlp.fpr()), np.array(mlp.tpr()))
		plt.title("AUC-ROC")
		plt.xlabel("FPR")
		plt.ylabel("TPR")
		plt.savefig(path + "auc-roc.png")

		#-----------------------------------------------------

	except Exception as AE:
		print(f"{type(AE).__name__}: {AE}", file=sys.stderr)
		exit(1)

	except KeyboardInterrupt:
		print("\nKeyboard Interuption", file=sys.stderr)

	print(f"model saved on 'data/{mlp.synthetic_name()}/mlp.npy'")

	return

if __name__ == "__main__":
	main()