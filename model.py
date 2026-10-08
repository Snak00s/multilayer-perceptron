import numpy as np
import random
from dataclasses import dataclass

class mlp_utils:

	def unison_shuffled_copies(a, b):
		assert len(a) == len(b)
		p = np.random.permutation(len(a))
		return a[p], b[p]

	def categoricalCrossentropy(predict, expPredict):
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
	activFunc: function

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
		self._delta = None

		return

	@classmethod
	def createLayer(cls, size: int, activation: str='sigmoid'):
		"""Create a layer with 'size' amount of nodes, this layer will use 'activation' as his activation function.
		- activation - selst one of ('sigmoid', 'softmax') or sigmoid by default.
		"""
		obj = cls()
		obj._size = size
		obj.nodes = [0 for _ in range(size)]
		match activation:
			case 'sigmoid':
				obj.activFunc = obj.sigmoid
			case 'softmax':
				obj.activFunc = obj.softmax
			case _:
				raise AssertionError(f"Invalid activation function: {activation}")
		return obj

	def sigmoid(self):
		self._activate = np.array(1 / (1 + np.exp(-self.nodes)))
		return self._activate

	def softmax(self):
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

	_layers: list[layer]
	_expectedOutput: list[float]
	_lossFunc: function

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
		"""Create a neural network with the given layers, 'layerList' should be a list of layer, the network design will not be modificable."""
		obj = cls()
		obj._layers = layerList
		obj._interLayerMatrix = [obj.__createInterLayerMatrix(layerList[x].size(), layerList[x + 1].size()) for x in range(len(layerList) - 1)]
		for lay in zip(range(len(obj._layers)), obj._layers):
			lay[1]._idx = lay[0]

		for lay in obj._layers:
			obj.__linkLayers(lay)
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
				layerInput = layer.prevLayer().activFunc()
				for i in range(self._sampleAmount):
					layer.nodes[i] = np.array([layerInput[i] @ layer.prevMatrix()[g] + 1 for g in range(layer.size())])
		self._layers[-1].activFunc()
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
					layerInput = layer.prevLayer().activFunc()
					for i in range(self.validSampleAmount):
						layer.nodes[i] = [(layerInput[i] @ layer.prevMatrix()[g]) + 1 for g in range(layer.size())]
			self.validLayers[-1].activFunc()
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

			self.validWeightMatrix = mod._interLayerMatrix
			tempLayers = mod.layers()
			self.validLayers = []
			for i in range(len(tempLayers)):
				self.validLayers.append(layer.createLayer(tempLayers[i].size(), tempLayers[i].activFunc.__name__))
			for lay in zip(range(len(self.validLayers)), self.validLayers):
				lay[1]._idx = lay[0]
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

	def __fillTrainIO(self, inputs: np.ndarray, expectedOutput: np.ndarray):
		assert len(inputs) == len(expectedOutput), "Amount of Inputs/Outputs missmatch"
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

	def fit(self,
	        trainData: tuple[np.ndarray],
			validData: tuple[np.ndarray],
			learningRate: float=0.0314,
			lossFunc: str='categoricalCrossentropy',
			batchSize: int=0,
			epoch: int=100,
			verbose: bool=False
		):
		"""Train the model on the given 'trainData=(tInputs, tOutputs)' and check validation with 'validData(vInputs, vOutputs)' for 'epoch' amount of time.
		For both, the len of inputs and outputs should be equal, also, trainData and validData should be differentfor coherent result.
		- learningRate - (0.00 < learningRate <= 1.00 ) - determine the 'speed' of the learning process, lower the value, lower the model learn, but better it learn.
		- lossFunc - 
		- bathSize - (0 <= batchSize <= len(inputs)) - change the way the model use the data, if 0, it will take the whole tInputs len as value.
		- epoch - (0 < epoch < inf) - determine the amount of time the model will train on the given data.
		- verbose - (True/False), show the evolution epoch per epoch.
		"""
		assert len(trainData[0]) == len(trainData[1]) and len(validData[0]) == len(validData[1]), "(Inputs, Outputs) len missmatch."
		assert trainData[0].shape and trainData[1].shape and validData[0].shape and validData[1].shape
		assert batchSize > 0 and learningRate > 0 and epoch > 0, "All integer value should be positive."

		match lossFunc:
			case 'categoricalCrossentropy':
				self._lossFunc = mlp_utils.categoricalCrossentropy
			case _:
				raise AssertionError(f'Invalid loss function: {lossFunc}')
		self.__fillTrainIO(trainData[0], trainData[1])
		self._learningRate = learningRate
		vMod = self.__validation(self, validData[0], validData[1])
		for i in range(epoch):
			self.__forwardPropagation()
			self._costMatrix = self._lossFunc(self._layers[-1].activate(), self._expectedOutput)
			self._costPerEpoch.append(mlp_utils.epochCost(self._costMatrix, self._expectedOutput))
			self._accuracy.append(mlp_utils.calculateAccuracy(self._layers[-1].activate(), self._expectedOutput))
			if (verbose is True):
				print(f"epoch {i + 1} / {epoch} | cost = {self._costPerEpoch[-1]:.4f} | accuracy = {self._accuracy[-1]:.4f}", end=' ')
			vMod.fwdPropagation()
			self._validCostMatrix = self._lossFunc(vMod.validLayers[-1].activate(), vMod.expectedOutput)
			self._validcostPerEpoch.append(mlp_utils.epochCost(self._validCostMatrix, vMod.expectedOutput))
			self._validAccuracy.append(mlp_utils.calculateAccuracy(vMod.validLayers[-1].activate(), vMod.expectedOutput))
			if (verbose is True):
				print(f"| validCost = {self._validcostPerEpoch[-1]:.4f} | validAccuracy = {self._validAccuracy[-1]:.4f}")
			self.__backPropagation()
			shufInputs = self._layers[0].nodes
			shufOutput = self._expectedOutput
			(self._layers[0].nodes, self._expectedOutput) = mlp_utils.unison_shuffled_copies(shufInputs, shufOutput)
			self._layers[0]._activate = self._layers[0].nodes
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