import numpy as np
from training.model import model


def main():

	data = np.load("data/mlp_data.npy", allow_pickle=True)

	testData = data.item()['test']

	mlp = np.load("data/mlp.npy", allow_pickle=True)

	mlp : model = mlp.item()

	res = mlp.useModel(testData)

	print(f"cost: {res[0]:.4f}, accuracy: {res[1]*100:.3f}%")

	return

if __name__ == "__main__":
	main()