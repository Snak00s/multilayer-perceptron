import numpy as np
import random
from dataclasses import dataclass

class mlp_utils:

	def unison_shuffled_copies(a, b):
		assert len(a) == len(b)
		p = np.random.permutation(len(a))
		return a[p], b[p]

	def catCrossEntropy(predict, expPredict):
		return np.array(-np.sum(expPredict * np.log(predict + 1e-12)))

	def	calculateAccuracy(predict: list[list], expPredict: list[list]):
		good = 0
		bad = 0
		for i in range(len(predict)):
			idx = False
			if (expPredict[i][1] > expPredict[i][0]):
				idx = True
			if (predict[i][int(idx)] > predict[i][int(not(idx))]):
				good += 1
			else:
				bad += 1
		return float(good / (good + bad))

	def epochCost(costMatrix, expOutput):
		return np.sum(costMatrix) / len(expOutput)

@dataclass
class layer:

	_prevLayer: layer
	_nextLayer: layer
	_delta: np.ndarray


	def __init__(self):
		self._idx = 0
		self._size = 0
		self.nodes = []

		self._prevMatrix = None
		self._nextMatrix = None

		self._prevLayer = None
		self._nextLayer = None

		self._activate = None

		self._sampleAmount = 0

		return

	@classmethod
	def createLayer(cls, idx: int, size: int):
		obj = cls()
		obj._idx = idx
		obj._size = size
		obj.nodes = [0 for _ in range(size)]
		return obj

	def sigmoid(self):
		self._activate = np.array(1 / (1 + np.exp(-self.nodes)))
		return self._activate

	def softMax(self):
		s = np.array([np.sum(np.exp(self.nodes[i])) for i in range(self._sampleAmount)])
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
		self._interLayerMatrix = []
		self._expectedOutput = []
		self._sampleAmount = 0
		self._learningRate = 0.0314

		self._costMatrix = []
		self._costPerEpoch = []
		self._accuracy = []

		self._validCostMatrix = []
		self._validcostPerEpoch = []
		self._validAccuracy = []
		return

	@classmethod
	def createNetwork(cls, layerList: list[layer]):
		obj = cls()
		obj._layers = layerList
		obj._interLayerMatrix = [obj.__createInterLayerMatrix(layerList[x].size(), layerList[x + 1].size()) for x in range(len(layerList) - 1)]
		for x in obj._layers:
			obj.__linkLayers(x)
		return obj

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

	def __delta(self, actualLayer: layer):
		if (actualLayer.nextLayer() == None):
			return np.array(actualLayer.activate() - self._expectedOutput)

		activate = actualLayer.activate()
		dA = activate * (1 - activate)
		delta = (self.__delta(actualLayer.nextLayer()) @ actualLayer.nextMatrix()) * dA

		return delta

	def __backPropagation(self):
		for layer in self._layers:
			if layer.prevLayer() is not None:
				layer._delta = self.__delta(layer)

		for layer in reversed(self._layers):
			if layer.prevLayer() is not None:
				# delta = self.__delta(layer)
				layer._prevMatrix -= (self._learningRate / self._sampleAmount * (layer._delta.T @ layer.prevLayer().activate()))
		return self

	def __forwardPropagation(self):
		for layer in self._layers:
			if layer.idx() == 0:
				continue
			if layer.idx() == 1:
				for i in range(self._sampleAmount):
					layerInput = layer.prevLayer().nodes[i]
					layer.nodes[i] = np.array([layerInput @ layer.prevMatrix()[j] + 1 for j in range(layer.size())])
			else:
				layerInput = layer.prevLayer().sigmoid()
				for i in range(self._sampleAmount):
					layer.nodes[i] = np.array([layerInput[i] @ layer.prevMatrix()[g] + 1 for g in range(layer.size())])
		self._layers[-1].softMax()
		return self

	class __validation:

		validLayers : list[layer]

		def fwdPropagation(self):
			for layer in self.validLayers:
				if layer.idx() == 0:
					continue
				if layer.idx() == 1:
					for i in range(self.validSampleAmount):
						layerInput = layer.prevLayer().nodes[i]
						layer.nodes[i] = [(layerInput @ layer.prevMatrix()[j]) + 1 for j in range(layer.size())]
				else:
					layerInput = layer.prevLayer().sigmoid()
					for i in range(self.validSampleAmount):
						layer.nodes[i] = [(layerInput[i] @ layer.prevMatrix()[g]) + 1 for g in range(layer.size())]
			self.validLayers[-1].softMax()
			return self

		def linkLayers(self, actualLayer: layer):
			if (actualLayer._idx > 0):
				actualLayer._prevMatrix = self.validWeightMatrix[actualLayer._idx - 1]
				actualLayer._prevLayer = self.validLayers[actualLayer._idx - 1]

			if (actualLayer._idx < len(self.validWeightMatrix)):
				actualLayer._nextMatrix = self.validWeightMatrix[actualLayer._idx]
				actualLayer._nextLayer = self.validLayers[actualLayer._idx + 1]
			return self

		def __init__(self, mod: model, inputs: list, expOutputs: list):

			assert len(inputs) == len(expOutputs)
			# for row in inputs:
			# 	assert len(row) == self.validLayers[0].size()
			# for row in expOutputs:
			# 	assert len(row) == self.validLayers[-1].size()

			self.validWeightMatrix = mod._interLayerMatrix

			tempLayers = mod.layers()

			self.validLayers = []
			for i in range(len(tempLayers)):
				self.validLayers.append(layer.createLayer(tempLayers[i].idx(), tempLayers[i].size()))
			for lay in self.validLayers:
				self.linkLayers(lay)

			inputs = np.array(inputs)
			self.validLayers[0].nodes = inputs
			self.validLayers[0]._activate = inputs.copy()
			self.validSampleAmount = len(inputs)

			for i in range(1, len(self.validLayers)):
				self.validLayers[i]._sampleAmount = self.validSampleAmount
				self.validLayers[i].nodes = np.zeros((self.validSampleAmount, self.validLayers[i].size()))

			self.expectedOutput = np.array(expOutputs)
			
			return

	def trainLoop(self, validInputs: list, validOutputs: list, epochAmount: int):

		assert len(validInputs) == len(validOutputs)
		for i in range(len(validInputs)):
			assert len(validInputs[i]) == len(validInputs[0]) and len(validOutputs[i]) == len(validOutputs[0])

		vMod = self.__validation(self, validInputs, validOutputs)
		for i in range(epochAmount):
			self.__forwardPropagation()
			self._costMatrix = mlp_utils.catCrossEntropy(self._layers[-1].activate(), self._expectedOutput)
			self._costPerEpoch.append(mlp_utils.epochCost(self._costMatrix, self._expectedOutput))
			self._accuracy.append(mlp_utils.calculateAccuracy(self._layers[-1].activate(), self._expectedOutput))
			print(f"epoch {i + 1} / {epochAmount} | cost = {self._costPerEpoch[-1]:.4f} | accuracy = {self._accuracy[-1]:.4f}", end=' ')

			vMod.fwdPropagation()

			self._validCostMatrix = mlp_utils.catCrossEntropy(vMod.validLayers[-1].activate(), vMod.expectedOutput)
			self._validcostPerEpoch.append(mlp_utils.epochCost(self._validCostMatrix, vMod.expectedOutput))
			print(f"| validCost = {self._validcostPerEpoch[-1]:.4f}")
			self._validAccuracy.append(mlp_utils.calculateAccuracy(vMod.validLayers[-1].activate(), vMod.expectedOutput))

			self.__backPropagation()

			shufInputs = self._layers[0].nodes
			shufOutput = self._expectedOutput
			(self._layers[0].nodes, self._expectedOutput) = mlp_utils.unison_shuffled_copies(shufInputs, shufOutput)
			self._layers[0]._activate = self._layers[0].nodes

		return self

	def fillTrainIO(self, inputs: list, expectedOutput: list):

		assert len(inputs) == len(expectedOutput)
		for row in inputs:
			assert len(row) == self._layers[0].size()
		for row in expectedOutput:
			assert len(row) == self._layers[-1].size()

		inputs = np.array(inputs)
		self._layers[0].nodes = inputs
		self._layers[0]._activate = inputs.copy()
		self._sampleAmount = len(inputs)
		for i in range(1, len(self._layers)):
			self._layers[i]._sampleAmount = self._sampleAmount
			self._layers[i].nodes = np.zeros((self._sampleAmount, self._layers[i].size()))

		self._expectedOutput = np.array(expectedOutput)
		return self

	def layers(self):
		return self._layers

	def costPerEpoch(self):
		return self._costPerEpoch

	def accuracy(self):
		return self._accuracy

	def validAccuracy(self):
		return self._validAccuracy

	def validCostPerEpoch(self):
		return self._validcostPerEpoch