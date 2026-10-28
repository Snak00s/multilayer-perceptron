import numpy as np
import pandas as pd
import sys
from pathlib import Path

from utils.parser import Parser

def help():
	print("this program will take a dataset as argument and slice it in 3 portion, train data, validation data and test data")
	print("\t--dataset:\tset the dataset to slice")
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

def count_occurence(arrayLike, to_find) -> int:

	c = int(0)
	for i in arrayLike:
		if type(i) == type(to_find) and i == to_find:
			c += 1
	return c

def main():
	try:
		parser = Parser()
		parser.special_help(help)
		parser.add_arg(argName="dataset", ammount=1, argType=str)

		args = parser.parse_args()

		assert len(args['dataset']) == 1

		data = pd.read_csv(args['dataset'][0], header=None).reset_index(drop="True")
		data_prcB = (count_occurence(data[1], 'B') / len(data[1])) * 100
		data = data.sample(frac=1)
		tLen = int(len(data) * 0.8)
		vLen = int(len(data) * 0.1)

		trainRows = data.iloc[0:tLen]
		tr_prcB = (count_occurence(trainRows[1], 'B') / len(trainRows[1])) * 100
		trainRows = np.array(data.iloc[0:tLen])

		trainInputs = np.array(trainRows[:, 2:], dtype=np.float64)
		trainInputs = (trainInputs - trainInputs.mean(axis=0)) / trainInputs.std(axis=0)
		trainOutputs = np.array(createExpectedOutput(trainRows[:, 1:2]))

		validRows = data.iloc[tLen:tLen + vLen]
		va_prcB = (count_occurence(validRows[1], 'B') / len(validRows[1])) * 100
		validRows = np.array(data.iloc[tLen:tLen + vLen])

		validInputs = np.array(validRows[:, 2:], dtype=np.float64)
		validInputs = (validInputs - validInputs.mean(axis=0)) / validInputs.std(axis=0)
		validOutputs = np.array(createExpectedOutput(validRows[:, 1:2]))

		testRows = data.iloc[tLen + vLen:]
		# te_prcB = (count_occurence(testRows[1], 'B') / len(testRows[1])) * 100
		testRows = np.array(data.iloc[tLen + vLen:])

		testInputs = np.array(testRows[:, 2:], dtype=np.float64)
		testInputs = (testInputs - testInputs.mean(axis=0)) / testInputs.std(axis=0)
		testOutputs = np.array(createExpectedOutput(testRows[:, 1:2]))

		arrDict = {
			'train': (trainInputs, trainOutputs),
			'valid': (validInputs, validOutputs),
			'test': (testInputs, testOutputs)
			}

		path = Path("data/mlp_data.npy")
		path.parent.mkdir(parents=True, exist_ok=True)
		np.save(path, arrDict)

		print(f"base data shape : {data.shape}	| {data_prcB:.2f}% 'B', {100 - data_prcB:.2f}% 'M' ")
		print(f"x_train shape : {trainInputs.shape}	| {tr_prcB:.2f}% 'B', {100 - tr_prcB:.2f}% 'M'")
		print(f"x_valid shape : {validInputs.shape}	| {va_prcB:.2f}% 'B', {100 - va_prcB:.2f}% 'M'")

	except Exception as AE:
		print(f"{type(AE).__name__}: {AE}", file=sys.stderr)
		exit(1)

	except KeyboardInterrupt:
		print("\nKeyboard Interuption", file=sys.stderr)

	return

if __name__ == "__main__":
	main()