import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

from bonus.modelBonus import model, layer, confMatrix
from utils.parser import Parser

def main():
	test = confMatrix(3)

	classes = [
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1]
		]

	expected = []
	predicted = []
	for _ in range(100):
		r = np.random.randint(0, 3)
		expected.append(classes[r])
		r = np.random.random(3)
		# normalized = np.zeros_like(r, dtype=int)
		# normalized[np.argmax(r)] = 1
		predicted.append(r)

	for i in range(len(predicted)):
		test.compute_result(predicted[i], expected[i])

	print(test.matrix)

	# print(np.delete(np.delete(test.matrix, 1, 0), 1, 1))
	print(test.TP())
	print(test.TN())
	print(test.FP())
	print(test.FN())

	return
	try:
		# -----------------------Parsing----------------------

		parser = Parser()

		parser.add_arg("layer", ammount=0, argType=int)
		parser.add_arg("epochs", ammount=1, argType=int)
		parser.add_arg("loss", ammount=1, argType=str)
		parser.add_arg("learning_rate", ammount=1, argType=float)
		parser.add_arg("batch_size", ammount=1, argType=int)
		parser.add_arg("optimizer", ammount=1, argType=str, optionnal=True)
		parser.add_arg("early_stop", ammount=1, argType=tuple, optionnal=True)

		args = parser.parse_args()

		#-----------------------Training----------------------

		data = np.load('data/mlp_data.npy', allow_pickle=True)

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

		model_path = Path("data/mlp.npy")
		model_path.parent.mkdir(parents=True, exist_ok=True)
		np.save(model_path, mlp)

		# pd.DataFrame({'Epochs': [i+1 for i in range(len(mlp.costPerEpoch()))],
		# 			  'Cost_T': mlp.costPerEpoch(),
		# 			  'Acc_T': mlp.accuracy(),
		# 			  'Cost_V': mlp.validCostPerEpoch(),
		# 			  'Acc_V': mlp.validAccuracy()}).set_index('Epochs').to_csv("modelsDataFrame.csv")

	except Exception as AE:
		print(f"{type(AE).__name__}: {AE}", file=sys.stderr)
		exit(1)

	except KeyboardInterrupt:
		print("\nKeyboard Interuption", file=sys.stderr)

	return

if __name__ == "__main__":
	main()

# 'model-CCE-GD-24-24-0.0314.cvs'