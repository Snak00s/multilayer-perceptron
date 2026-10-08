import sys

class Parser:
	"""This is a sys.argv parser, user should use 'add_arg()' after construction then 'parse_arg()'"""

	_argDict: dict[str, list]
	_argNarg: dict[str, int]
	_argType: dict[str, type[int]]

	def __init__(self):
		self._argDict = dict()
		self._argNarg = dict()
		self._argType = dict()
		pass

	def add_arg(self, argName: str, ammount: int, argType: type[type]):
		"""Add an argument to the list of argument to catch.
		- argName - add the argument name to manage DO NOT PUT '-' character at begin or end.
		- ammount - define the ammount of value per arguemnt if <= 0 : will set to sys.maxsize.
		- argType - define the value type of the argument.
		"""
		if (self._argDict.get(argName) is not None):
			raise AssertionError(f"Argument {argName}: Already set")
		self._argDict[argName] = []
		self._argNarg[argName] = ammount if ammount != 0 else sys.maxsize
		self._argType[argName] = argType
		return self

	def parse_args(self):
		"""Will parse argument in sys.argv,
		it should catch all argument added via 'add_arg()' and respect given limitation"""
		name = ''
		obj = int

		for val in sys.argv[1:]:
			if self._argDict.get(val.strip('-')) is not None:
				name = val.strip('-')
				obj = self._argType[name]
				continue

			if len(self._argDict[name]) < self._argNarg[name]:
				self._argDict[name].append(obj(val))
			else:
				raise AssertionError(f"{name}: too many value, max={self._argNarg[name]}.")

		for arg in self._argDict.keys():
			if (len(self._argDict[arg]) < self._argNarg[arg] and self._argNarg[arg] != sys.maxsize):
				raise AssertionError(f"{arg}: not enought argument, nargs={self._argNarg[arg]}")
			
		return self._argDict

	def get_argument_status(self, argName: str):
		return [self._argDict[argName],
			(len(self._argDict[argName]), self._argNarg[argName]),
			self._argType[argName]
			]


# def main():
# 	try:
# 		test = Parser()
# 		test.add_arg('epochs', ammount=3, argType=int)
# 		test.add_arg('layer', ammount=3, argType=int)
# 		test.add_arg('learningRate', ammount=1, argType=float)
# 		res = test.parse_args()
# 		print(res['epochs'])
# 	except Exception as AE:
# 		print(f"{type(AE).__name__}: {AE}", file=sys.stderr)
# 		exit(1)

# if __name__ == "__main__":
# 	main()