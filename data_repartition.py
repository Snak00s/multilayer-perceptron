import numpy as np
import pandas as pd
import random
import sys
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
		parser = Parser()
		parser.add_arg(argName="dataset", ammount=1, argType=str)

		args = parser.parse_args()

		assert len(args['dataset']) == 1

		data = pd.read_csv(args['dataset'][0], header=None).reset_index(drop="True")
		data = data.sample(frac=1)
		tLen = int(len(data) * 0.8)
		vLen = int(len(data) * 0.1)
		dataRows = np.array(data.iloc[0:tLen])
		trainInputs = np.array(dataRows[:, 2:], dtype=np.float64)
		trainInputs = (trainInputs - trainInputs.mean(axis=0)) / trainInputs.std(axis=0)
		trainOutputs = np.array(createExpectedOutput(dataRows[:, 1:2]))

		validRows = np.array(data.iloc[tLen:tLen + vLen])
		validInputs = np.array(validRows[:, 2:], dtype=np.float64)
		validInputs = (validInputs - validInputs.mean(axis=0)) / validInputs.std(axis=0)
		validOutputs = np.array(createExpectedOutput(validRows[:, 1:2]))

		testRows = np.array(data.iloc[tLen + vLen:])
		testInputs = np.array(testRows[:, 2:], dtype=np.float64)
		testInputs = (testInputs - testInputs.mean(axis=0)) / testInputs.std(axis=0)
		testOutputs = np.array(createExpectedOutput(testRows[:, 1:2]))

		arrDict = {
			'train': (trainInputs, trainOutputs),
			'valid': (validInputs, validOutputs),
			'test': (testInputs, testOutputs)
			}

		np.save('mlp_data', arrDict)

		print(f"x_train shape : {trainInputs.shape}")
		print(f"x_valid shape : {validInputs.shape}")

	except Exception as AE:
		print(f"{type(AE).__name__}: {AE}", file=sys.stderr)
		exit(1)

	return

if __name__ == "__main__":
	main()