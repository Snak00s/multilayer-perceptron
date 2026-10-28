import numpy as np
import matplotlib.pyplot as plt
import sys
from utils.parser import Parser
import os
from bonus.modelBonus import model

def help():
	print("model comparator, will take all model in the given directory and plot their metrics in a shared graph.")
	print("\t--dir:\tset the dir to find all models")
	return

def plot_data(to_plot: list, legend: list[str], x_label: str, y_label: str, title: str, name: str):
	plt.figure()
	for arg in to_plot:
		plt.plot(arg)
	plt.legend(legend)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.savefig(name)
	return

def plot_roc(x_plot: list, y_plot: list, legend: list[str], x_label: str, y_label: str, title: str, name: str):
	plt.figure()
	for arg in zip(x_plot, y_plot):
		plt.plot(arg[0], arg[1])
	plt.legend(legend)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.savefig(name)
	return

def main():
	try:
		parser = Parser()
		parser.special_help(help)
		parser.add_arg("dir", ammount=1, argType=str)

		args = parser.parse_args()
		models_dir = args["dir"][0] + "/"
		tmp: list[str] = os.listdir(models_dir)

		models_lst: list[model] = []
		for d in tmp:
			if d.startswith('model-') and os.path.isdir(models_dir + d) is True:
				obj = np.load(models_dir + d + "/mlp.npy", allow_pickle=True)
				obj = obj.item()
				models_lst.append(obj)

		models_trainCost = []
		models_validCost = []
		models_trainAccuracy = []
		models_validAccuracy = []
		models_precision = []
		models_recall = []
		models_f1_score = []
		models_tpr = []
		models_fpr = []
		models_name = []
		for mod in models_lst:
			models_trainCost.append(mod.costPerEpoch())
			models_validCost.append(mod.validCostPerEpoch())
			models_trainAccuracy.append(mod.accuracy())
			models_validAccuracy.append(mod.validAccuracy())
			models_precision.append(mod.precision())
			models_recall.append(mod.recall())
			models_f1_score.append(mod.f1_score())
			models_tpr.append(mod.tpr())
			models_fpr.append(mod.fpr())
			models_name.append(mod.synthetic_name())
		plot_data(models_trainCost, models_name, "epoch", "cost", "TrainCost", "traincost.png")
		plot_data(models_validCost, models_name, "epoch", "cost", "ValidCost", "validcost.png")
		plot_data(models_trainAccuracy, models_name, "epoch", "accuracy", "TrainAccuracy", "trainaccuracy.png")
		plot_data(models_validAccuracy, models_name, "epoch", "accuracy", "ValidAccuracy", "validaccuracy.png")
		plot_data(models_precision, models_name, "epoch", "precision", "Precision", "precision.png")
		plot_data(models_recall, models_name, "epoch", "recall", "Recall", "recall.png")
		plot_data(models_f1_score, models_name, "epoch", "f1_score", "F1_score", "f1_score.png")
		plot_roc(models_fpr, models_tpr, models_name, "fpr", "tpr", "AUC-ROC", "auc-roc.png")
			

	except Exception as AE:
		print(f"{type(AE).__name__}: {AE}", file=sys.stderr)
		exit(1)
	except KeyboardInterrupt:
		print("\nKeyboard Interuption", file=sys.stderr)

if __name__ == "__main__":
	main()