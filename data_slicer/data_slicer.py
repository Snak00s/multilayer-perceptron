import numpy as np
import pandas as pd
import sys
from pathlib import Path

from utils.parser import Parser

def help():
	print("this program will take a dataset as argument and slice it in 2 portion, train data, validation data and test data")
	print("\t--dataset:\tset the dataset to slice")
	return

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

		trainRows = data.iloc[0:tLen].assign(e='train')
		tr_prcB = (count_occurence(trainRows[1], 'B') / len(trainRows[1])) * 100

		validRows = data.iloc[tLen:].assign(e='valid')
		va_prcB = (count_occurence(validRows[1], 'B') / len(validRows[1])) * 100

		data_training = pd.concat([trainRows, validRows]).reset_index(drop=True)

		path = Path("data/data_training.csv")
		path.parent.mkdir(parents=True, exist_ok=True)
		data_training.to_csv(path, index=False, header=False)

		print(f"base data shape : {data_training.shape}	| {data_prcB:.2f}% 'B', {100 - data_prcB:.2f}% 'M' ")
		print(f"x_train shape : {trainRows.shape}	| {tr_prcB:.2f}% 'B', {100 - tr_prcB:.2f}% 'M'")
		print(f"x_valid shape : {validRows.shape}	| {va_prcB:.2f}% 'B', {100 - va_prcB:.2f}% 'M'")

	except Exception as AE:
		print(f"{type(AE).__name__}: {AE}", file=sys.stderr)
		exit(1)

	except KeyboardInterrupt:
		print("\nKeyboard Interuption", file=sys.stderr)

	return

if __name__ == "__main__":
	main()