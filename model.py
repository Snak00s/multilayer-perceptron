import numpy as np
import random
from dataclasses import dataclass, field

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

		self._activate = None

		self._inputIdx = 0
		self._sampleAmount = 0

		return

	@classmethod
	def createLayer(cls, idx: int, size: int):
		obj = cls()
		obj._idx = idx
		obj._size = size
		obj.nodes = [0 for _ in range(size)]
		return obj

	def aplySigmoid(self):
		self._activate = np.array([[1 / (1 + np.exp(-self.nodes[i][x])) for x in range(self._size)] for i in range(self._sampleAmount)])
		return self._activate

	def softMax(self):
		s = [np.sum([np.exp(x) for x in self.nodes[i]]) for i in range(self._sampleAmount)]
		self._activate = np.array([[np.exp(x) / s[i] for x in self.nodes[i]] for i in range(self._sampleAmount)])
		return self._activate

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

	def activate(self):
		return self._activate

class model:

	_layers : list[layer]
	_expectedOutput : list[float]

	def __init__(self):
		self._layers = []
		self._interLayerMatrix = None
		self._expectedOutput = []
		self._inputIdx = 0
		self._sampleAmount = 0
		self._costMatrix = []
		self._learningRate = 0.01
		self._costPerEpoch = []
		self._accuracy = []
		return

	def __delta(self, actualLayer: layer):
		if (actualLayer.nextLayer() == None):
			return np.array(actualLayer.activate() - self._expectedOutput)

		activate = actualLayer.activate()
		dA = activate * (1 - activate)
		delta = (self.__delta(actualLayer.nextLayer()) @ actualLayer.nextMatrix()) * dA

		return delta

	def __backPropagation(self):
		for layer in reversed(self._layers):
			if layer.nextLayer() == None:
				delta = self.__delta(layer)
				layer._prevMatrix = layer.prevMatrix() - (self._learningRate * (1 / self._sampleAmount) * np.dot(delta.T, layer.prevLayer().activate()))
			elif layer.prevLayer() != None:
				delta = self.__delta(layer)
				# layer._prevMatrix = layer.prevMatrix() - (self._learningRate * (1 / self._sampleAmount) * np.dot(delta.T, layer.prevLayer().activate()))
				layer._prevMatrix = layer.prevMatrix() - (self._learningRate * np.dot(delta.T, layer.prevLayer().activate()))
		return self

	def __catCrossEntropy(self):
		predict = self.layers()[-1].activate()
		# self._costMatrix = np.array(-((self._expectedOutput * np.log(predict)) + (1 - self._expectedOutput) * np.log(1 - predict)))
		# self._costMatrix = np.array(-np.sum(self._expectedOutput * np.log(predict)))
		return np.array(-np.sum(self._expectedOutput * np.log(predict)))

	def __forwardPropagation(self):
		for layer in self._layers:
			if layer.idx() == 0:
				continue
			if layer.idx() == 1:
				for i in range(self._sampleAmount):
					layerInput = layer.prevLayer().nodes[i]
					layer.nodes[i] = [np.dot(layerInput, layer.prevMatrix()[j]) + 1 for j in range(layer.size())]
			else:
				layerInput = layer.prevLayer().aplySigmoid()
				for i in range(self._sampleAmount):
					layer.nodes[i] = [np.dot(layerInput[i], layer.prevMatrix()[g]) + 1 for g in range(layer.size())]
		self._layers[-1].softMax()
		return self

	def	__calculateAccuracy(self):
		good = 0
		bad = 0
		output = self._layers[-1].activate()
		for i in range(len(output)):
			idx = False
			if (self._expectedOutput[i][1] > self._expectedOutput[i][0]):
				idx = True
			if (output[i][int(idx)] > output[i][int(not(idx))]):
				good += 1
			else:
				bad += 1
		return float(good / (good + bad))

	def trainLoop(self, validInputs: list, validOutputs: list, epochAmount: int):
		for i in range(epochAmount):
			self.__forwardPropagation()
			self._costMatrix = self.__catCrossEntropy()
			print(f"epoch {i + 1} / {epochAmount} | cost =", self.__epochCost())
			self._costPerEpoch.append(self.__epochCost())
			self._accuracy.append(self.__calculateAccuracy())

			self.__backPropagation()
		return self

	def validation(self, inputs: list, expectedOutput: list):
		self.fillInputsLayer(inputs)
		self.fillExpectedOutput(expectedOutput)
		self.__forwardPropagation()
		print(f"| validCost =", self.__epochCost())
		return self

	def fillInputsLayer(self, inputs: list):
		for row in inputs:
			assert len(row) == self._layers[0].size()
		self._layers[0].nodes = inputs
		self._layers[0]._activate = inputs.copy()
		self._sampleAmount = len(inputs)
		row = len(row)
		for i in range(1, len(self._layers)):
			self._layers[i]._sampleAmount = self._sampleAmount
			self._layers[i].nodes = np.zeros((self._sampleAmount, self._layers[i].size()))
		return self

	def fillExpectedOutput(self, expectedOutput):
		for row in expectedOutput:
			assert len(row) == self._layers[-1].size()
		self._expectedOutput = np.array(expectedOutput)
		return self

	@staticmethod
	def __createInterLayerMatrix(prevLayerSize: int, actualLayerSize: int):
		upLim = np.sqrt(6 / prevLayerSize)
		downLim = - np.sqrt(6 / prevLayerSize)
		return np.array([[random.uniform(downLim, upLim) for _ in range(prevLayerSize)] for _ in range(actualLayerSize)])

	def __linkLayers(self, actualLayer: layer):
		if (actualLayer._idx > 0):
			actualLayer._prevMatrix = self._interLayerMatrix[actualLayer._idx - 1]
			actualLayer._prevLayer = self._layers[actualLayer._idx - 1]

		if (actualLayer._idx < len(self._interLayerMatrix)):
			actualLayer._nextMatrix = self._interLayerMatrix[actualLayer._idx]
			actualLayer._nextLayer = self._layers[actualLayer._idx + 1]
		return self

	@classmethod
	def createNetwork(cls, lst: list[layer]):
		obj = cls()
		obj._layers = lst
		obj._interLayerMatrix = [obj.__createInterLayerMatrix(lst[x].size(), lst[x + 1].size()) for x in range(len(lst) - 1)]
		for x in obj._layers:
			obj.__linkLayers(x)
		return obj

	def layers(self):
		return self._layers

	def interLayerMatrix(self):
		return self._interLayerMatrix

	def __epochCost(self):
		return np.sum(self._costMatrix) / len(self._expectedOutput)

	def resetLayers(self):
		for i in range(len(self._layers)):
			self._layers[i].nodes = []
			self._layers[i]._activate = []
		return self

	def costPerEpoch(self):
		return self._costPerEpoch

	def accuracy(self):
		return self._accuracy