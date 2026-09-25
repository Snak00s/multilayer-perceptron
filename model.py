import numpy as np
import random
import math
from dataclasses import dataclass, field

# def sigNumber(x, n):
# 	if x == 0:
# 		return 0
# 	return round(x, n - 1 - int(math.floor(math.log10(abs(x)))))

# def weightDot(input_lst: list, weight_lst: list):
# 	"""Calculate the node value before the activation func"""
# 	return np.dot(input_lst, weight_lst) + 1

@dataclass
class layer:

	_prevLayer: layer
	_nextLayer: layer

	def __init__(self):
		self._idx = 0
		self._size = 0
		self.nodes = []

		self._prevMatrix = None
		self._nextMatrix = None

		self._prevLayer = None
		self._nextLayer = None

		return

	@classmethod
	def createLayer(cls, idx: int, size: int):
		obj = cls()
		obj._idx = idx
		obj._size = size
		obj.nodes = [0 for _ in range(size)]
		return obj

	def sigmoid(self) -> list:
		return [1 / (1 + np.exp(-self.nodes[x])) for x in range(self._size)]

	def idx(self):
		return self._idx

	def size(self):
		return self._size

	def prevLayer(self):
		return self._prevLayer

	def nextLayer(self):
		return self._nextLayer

	def prevMatrix(self):
		return self._prevMatrix

	def nextMatrix(self):
		return self._nextMatrix

class model:

	_layers : list[layer]
	_expectedOutput : list[float]

	def __init__(self):
		self._layers = []
		self._interLayerMatrix = []
		self._expectedOutput = []
		return

	@staticmethod
	def createInterLayerMatrix(prevLayerSize: int, actualLayerSize: int) -> list :

		upLim = np.sqrt(6 / prevLayerSize)
		downLim = - np.sqrt(6 / prevLayerSize)
		return [[random.uniform(downLim, upLim) for _ in range(prevLayerSize)] for _ in range(actualLayerSize)]

	def linkLayers(self, actualLayer: layer):

		if (actualLayer._idx > 0):
			actualLayer._prevMatrix = self._interLayerMatrix[actualLayer._idx - 1]
			actualLayer._prevLayer = self._layers[actualLayer._idx - 1]

		if (actualLayer._idx < len(self._interLayerMatrix)):
			actualLayer._nextMatrix = self._interLayerMatrix[actualLayer._idx]
			actualLayer._nextLayer = self._layers[actualLayer._idx]
		return self

	def fillInputsLayer(self, inputs: list):
		try:
			assert len(inputs) == self._layers[0].size()
		except AssertionError:
			print("Error: model.fillInputsLayer: len(inputs) != _layers[0].size()")
			exit(1)
		self._layers[0].nodes = inputs
		return self

	def fillExpectedOutput(self, expectedOutput):
		try:
			assert len(expectedOutput) == self._layers[-1].size()
		except AssertionError:
			print("Error: model.fillInputsLayer: len(inputs) != _layers[0].size()")
			exit(1)
		self._expectedOutput = expectedOutput
		return self

	def forwardPropagation(self):
		for layer in self._layers:
			if layer.idx() != 0:
				activateValue = layer.prevLayer().sigmoid()
				layer.nodes = [np.dot(activateValue, layer.prevMatrix()[i]) + 1 for i in range(layer.size())]
		return self

	# def backPropagation(self):
	# 	for layer in reversed(self._layers):
	# 		matrix = layer.prevMatrix()
	# 		if (matrix != None):
	# 			for i in range(len(matrix)):
	# 				for j in range(len(matrix[i])):
	# 	return

	@classmethod
	def createNetwork(cls, lst: list[layer]):
		obj = cls()
		obj._layers = lst
		obj._interLayerMatrix = [obj.createInterLayerMatrix(lst[x].size(), lst[x + 1].size()) for x in range(len(lst) - 1)]
		for x in obj._layers:
			obj.linkLayers(x)
		return obj

	def layers(self):
		return self._layers

	def interLayerMatrix(self):
		return self._interLayerMatrix