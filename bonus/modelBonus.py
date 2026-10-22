import numpy as np
import random
from dataclasses import dataclass
import sys

class confMatrix:

	matrix: np.ndarray
	classList: np.ndarray

	def __init__(self, clas: int):
		self.clas = clas
		self.matrix = np.zeros((clas, clas), dtype=int)
		self.classList = np.identity(clas, dtype=int)

	def __compute_to_class(self, matrix):
		arr = np.array(matrix)
		normalized = np.zeros_like(arr)
		normalized[np.argmax(arr)] = 1

		idx = np.where(np.all(self.classList==normalized, axis=1))
		
		return idx

	def compute_result(self, predicted, expected):
		class_e = self.__compute_to_class(expected)
		class_p = self.__compute_to_class(predicted)

		self.matrix[class_e, class_p] += 1
		return

	#True Positive ------------------------------------------------
	def	TP_class(self, i: int):
		return self.matrix[i][i]

	def TP(self):
		return np.sum([self.TP_class(i) for i in range(self.clas)])

	#False Negative -----------------------------------------------
	def FN_class(self, i: int):
		return np.sum(self.matrix[i]) - self.TP_class(i)

	def FN(self):
		return np.sum([self.FN_class(i) for i in range(self.clas)])

	#False Positive -----------------------------------------------
	def FP_class(self, i):
		return np.sum(self.matrix[:, i]) - self.TP_class(i)

	def FP(self):
		return np.sum([self.FP_class(i) for i in range(self.clas)])

	#True Negative ------------------------------------------------
	def TN_class(self, i):
		tn = np.delete(self.matrix, i, 0)
		tn = np.delete(tn, i, 1)
		return np.sum(tn)

	def TN(self):
		return np.sum([self.TN_class(i) for i in range(self.clas)])


class routine_model:
	def __init__(self):
		pass

	def routine(self):
		pass

class mlp_utils:

	def unison_shuffled_copies(a, b):
		"""
		Shuffle a and b by respect of their correspondind index
		"""
		assert len(a) == len(b)
		p = np.random.permutation(len(a))
		return a[p], b[p]

	def categoricalCrossentropy(predict, expPredict) -> float:
		"""
		Calculate the loss on the given 'predict' + 'expPredict' combination
		"""
		return -np.sum(expPredict * np.log(predict + 1e-12))

	def	calculateAccuracy(predict: list[list], expPredict: list[list]):
		"""
		Calculate accuray on the given 'predict' + 'expPredict' combination
		"""
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

	def batchCost(costMatrix, expOutput):
		return costMatrix / len(expOutput)

	def ponderate(matrix, samplePerBatch):
		"""
		Will ponderate the matrix value depending on the batchSize used, usefull when a dataset size is not a multiplicative of the batchSize used
		"""
		return np.sum([matrix[i] * samplePerBatch[i] for i in range(len(samplePerBatch))]) / np.sum(samplePerBatch)

class _NesMomentum:
	_dirWeight: list[np.ndarray]
	_weightVelocity: list[np.ndarray]

	_dirBias: list[np.ndarray]
	_biasVelocity: list[np.ndarray]

	_factor: float
	_lr: float

	_layers: list[layer]
	_weightMatrixs: list[np.ndarray]
	_biasMatrixs: list[np.ndarray]
	_refMod: model

	def __init__(self, mod: model, factor: float):
		self._refMod = mod
		self._lr = mod._learningRate
		self._factor = factor

		self._layers = mod._layers
		self._weightMatrixs = mod._weightMatrixs
		self._biasMatrixs = mod._biasMatrixs

		self._dirWeight = [weight.copy() for weight in mod._weightMatrixs]
		self._weightVelocity = [np.zeros(self._dirWeight[i].shape, dtype=np.float64) for i in range(len(self._dirWeight))]

		self._dirBias = [mod.layers()[i].bias.copy() for i in range(1,len(mod.layers()))]
		self._biasVelocity = [np.zeros(self._dirBias[i].shape, dtype=np.float64) for i in range(len(self._dirBias))]

		self.linkLayersToDir()

		return

	def linkLayersToDir(self):
		for i in range(1, len(self._layers)):
			self._layers[i]._weightMatrix = self._dirWeight[i-1]
			self._layers[i].bias = self._dirBias[i-1]
		return

	def update_dirParam(self):
		for i in range(len(self._dirWeight)):
			self._dirWeight[i] = self._weightMatrixs[i] - self._factor * self._weightVelocity[i] #maybe +
			self._layers[i+1]._weightMatrix = self._dirWeight[i]

			self._dirBias[i] = self._biasMatrixs[i] - self._factor * self._biasVelocity[i]
			self._layers[i+1].bias = self._dirBias[i]
		return

	def update_velocity(self, batchSize: int):
		for i in range(len(self._weightVelocity)):
			self._weightVelocity[i] *= self._factor
			self._weightVelocity[i] += self._lr * self._layers[i+1]._wGrad #maybe -

			self._biasVelocity[i] *= self._factor
			self._biasVelocity[i] += self._lr * self._layers[i+1]._bGrad
		return

	def update_trueParam(self):
		for i in range(len(self._weightMatrixs)):
			self._weightMatrixs[i] -= self._weightVelocity[i] #maybe +
			self._biasMatrixs[i] -= self._biasVelocity[i]

			self._layers[i+1]._weightMatrix = self._weightMatrixs[i]
			self._layers[i+1].bias = self._biasMatrixs[i]
		return

	def routine(self):
		self.update_dirParam()
		(co, acc) = self._refMod.routine()
		self.update_velocity(self._refMod._sampleAmount)
		self.update_trueParam()
		return (co, acc)

class _validation:

	validLayers : list[layer]

	def fwdPropagation(self):
		for layer in self.validLayers:
			if layer.idx() == 0:
				continue
			if layer.idx() == 1:
				for i in range(self.validSampleAmount):
					layerInput = layer.prevLayer().nodes[i]
					layer.nodes[i] = np.array([(layerInput @ layer.weightMatrix()[j]) + layer.bias[j] for j in range(layer.size())])
			else:
				layerInput = layer.prevLayer()._activFunc()
				for i in range(self.validSampleAmount):
					layer.nodes[i] = np.array([(layerInput[i] @ layer.weightMatrix()[g]) + layer.bias[g] for g in range(layer.size())])
		self.validLayers[-1]._activFunc()
		return self

	def linkLayers(self, actualLayer: layer):
		if (actualLayer._idx > 0):
			actualLayer._weightMatrix = self.validWeightMatrix[actualLayer._idx - 1]
			actualLayer._prevLayer = self.validLayers[actualLayer._idx - 1]
		if (actualLayer._idx < len(self.validWeightMatrix)):
			actualLayer._nextLayer = self.validLayers[actualLayer._idx + 1]
		return self

	def updateBias(self, layers: list[layer]) -> None:
		for i in range(len(self.validLayers)):
			self.validLayers[i].bias = layers[i].bias
		return

	def __init__(self, mod: model, inputs: list, expOutputs: list):

		self.validWeightMatrix = mod._weightMatrixs
		tempLayers = mod.layers()
		self.validLayers = []
		for i in range(len(tempLayers)):	# copy the layer configuration of the main model
			self.validLayers.append(layer.createLayer(tempLayers[i].size(), tempLayers[i]._activFunc.__name__))
			self.validLayers[i]._idx = i
			self.validLayers[i].bias = tempLayers[i].bias
			if (i > 0):
				self.linkLayers(self.validLayers[i - 1])
		self.linkLayers(self.validLayers[-1])
		inputs = np.array(inputs)
		self.validLayers[0].nodes = inputs
		self.validLayers[0]._activate = inputs.copy()
		self.validSampleAmount = len(inputs)

		for i in range(1, len(self.validLayers)):	#initialze all layers
			self.validLayers[i]._sampleAmount = self.validSampleAmount
			self.validLayers[i].nodes = np.zeros((self.validSampleAmount, self.validLayers[i].size()))
		self.expectedOutput = np.array(expOutputs)
		return

@dataclass
class layer:

	_prevLayer: layer
	_nextLayer: layer
	_delta: np.ndarray
	_activFunc: function

	def __init__(self):
		self._idx = 0
		self._size = 0
		self.nodes = np.array([], dtype=np.float64)
		self.bias = np.array([], dtype=np.float64)
		self._weightMatrix = np.array([], dtype=np.float64)
		self._prevLayer = None
		self._nextLayer = None
		self._activate = None
		self._sampleAmount = 0
		self._delta = None
		self._wGrad = None
		self._bGrad = None
		return

	@classmethod
	def createLayer(cls, size: int, activation: str='sigmoid'):
		"""Create a layer with 'size' amount of nodes, this layer will use 'activation' as his activation function.
		- activation - selst one of ('sigmoid', 'softmax') or sigmoid by default.
		"""
		obj = cls()
		obj._size = size
		obj.nodes = np.array([0 for _ in range(size)], dtype=np.float64)
		match activation:
			case 'sigmoid':
				obj._activFunc = obj.sigmoid
			case 'softmax':
				obj._activFunc = obj.softmax
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

	def weightMatrix(self):
		return self._weightMatrix

	def activate(self):
		return self._activate

class model:

	_layers: list[layer]
	_expectedOutput: list[float]
	_trainInputs: list[float]
	_lossFunc: function
	_weightMatrixs: list[np.ndarray]

	def __init__(self):
		self._layers = []
		self._weightMatrixs = []
		self._biasMatrixs = []
		self._expectedOutput = []
		self._trainOutputs = []
		self._trainInputs = []
		self._sampleAmount = 0
		self._learningRate = 0
		self._costPerEpoch = []
		self._accuracy = []
		self._validCost = 0
		self._validcostPerEpoch = []
		self._validAccuracy = []

		#Early Stopping
		self._bestCost = sys.maxsize
		self._patience = 0
		self._minEcart = 0.0
		return

	@classmethod
	def createNetwork(cls, layerList: list[layer]):
		"""Create a neural network with the given layers, 'layerList' should be a list of layer, the network design will not be modificable."""
		obj = cls()
		obj._layers = layerList
		obj._weightMatrixs = [obj.__createInterLayerMatrix(layerList[x].size(), layerList[x + 1].size()) for x in range(len(layerList) - 1)]
		obj._biasMatrixs = [np.array([1 for _ in range(layerList[x].size())], dtype=np.float64) for x in range(1, len(layerList))]
		for lay in zip(range(len(obj._layers)), obj._layers):
			lay[1]._idx = lay[0]

		for lay in obj._layers:
			obj.__linkLayers(lay, obj._weightMatrixs, obj._biasMatrixs)
		obj._lossFunc = mlp_utils.categoricalCrossentropy
		return obj

	@staticmethod
	def __createInterLayerMatrix(prevLayerSize: int, actualLayerSize: int):
		upLim = np.sqrt(6 / prevLayerSize)
		downLim = - np.sqrt(6 / prevLayerSize)
		return np.array([[random.uniform(downLim, upLim) for _ in range(prevLayerSize)] for _ in range(actualLayerSize)])

	def __linkLayers(self, actualLayer: layer, weightMatrix: list, biasMatrix: list):
		if (actualLayer._idx > 0):
			actualLayer._weightMatrix = weightMatrix[actualLayer._idx - 1]
			actualLayer.bias = biasMatrix[actualLayer._idx - 1]
			actualLayer._prevLayer = self._layers[actualLayer._idx - 1]
		if (actualLayer._idx < len(weightMatrix)):
			actualLayer._nextLayer = self._layers[actualLayer._idx + 1]
		return self

	def _delta(self, actualLayer: layer): 
		if (actualLayer.nextLayer() == None):
			return np.array(actualLayer.activate() - self._expectedOutput)
		activate = actualLayer.activate()
		dA = activate * (1 - activate)
		delta = (actualLayer.nextLayer()._delta @ actualLayer.nextLayer().weightMatrix() * dA)

		return delta

	def backPropagation(self):
		for layer in reversed(self._layers):
			layer._delta = self._delta(layer)
		for layer in reversed(self._layers):
			if layer.prevLayer() is not None:
				layer._wGrad = layer._delta.T @ layer.prevLayer().activate() / self._sampleAmount
				layer._weightMatrix -= self._learningRate * layer._wGrad

				layer._bGrad = np.sum(layer._delta, axis=0) / self._sampleAmount
				layer.bias -= self._learningRate * layer._bGrad
		return

	def forwardPropagation(self):
		for lay in self._layers:
			if lay.idx() == 0:	#should not apply any activation on the input layer
				continue
			layerInput = lay.prevLayer().nodes if lay.idx() == 1 else lay.prevLayer()._activFunc()
			for i in range(self._sampleAmount):
				lay.nodes[i] = np.array([layerInput[i] @ lay.weightMatrix()[g] + lay.bias[g] for g in range(lay.size())])
		self._layers[-1]._activFunc()
		return

	def routine(self):
			self.forwardPropagation()					#feedforward
			costMatrix = self._lossFunc(self._layers[-1].activate(), self._expectedOutput)
			self.backPropagation()						#bacpropagation to learn on every batch
			return (mlp_utils.batchCost(costMatrix, self._expectedOutput),
					mlp_utils.calculateAccuracy(self._layers[-1].activate(), self._expectedOutput))

	def __fillTrainIO(self, inputs: np.ndarray, expectedOutput: np.ndarray):
		assert len(inputs) == len(expectedOutput), "Amount of Inputs/Outputs missmatch"
		for row in inputs:
			assert len(row) == self._layers[0].size()
		for row in expectedOutput:
			assert len(row) == self._layers[-1].size()
		self._trainInputs = np.array(inputs)
		self._trainOutputs = np.array(expectedOutput)
		return self

	def __updateNodes(self, step: int, batchSize: int): #because batch used are not always the same size we need to adapt the shape of the nodes for every batch

		inputSample = self._trainInputs[step:step + batchSize]
		self._sampleAmount = len(inputSample)
		for i in range(len(self._layers)):
			self._layers[i]._sampleAmount = self._sampleAmount
			if i != 0:
				self._layers[i].nodes = np.zeros((self._sampleAmount, self._layers[i].size())) #nodes should be reset because their value are not important for the learning
			else:
				self._layers[i].nodes = inputSample
				self._layers[i]._activate = inputSample.copy()
		self._expectedOutput = self._trainOutputs[step:step + batchSize]
		if (step + self._sampleAmount == len(self._trainInputs)):
			return -1
		return step + self._sampleAmount

	def _earlyStopping(self):
		if self._validcostPerEpoch[-1] < self._bestCost - self._minEcart:
			self._bestCost = self._validcostPerEpoch[-1]
			self._patience = 0
		else:
			self._patience += 1

		return self._patience

	###################
	def fit(self,
	        trainData: tuple[np.ndarray],
			validData: tuple[np.ndarray],
			learningRate: float,
			lossFunc: str,
			batchSize: int,
			epoch: int,
			optimizer: str='',
			verbose: bool=False,
			earlyStopping: tuple=(int(0), float(0)) # tuple['patience': int, 'minEcart':float]
		):
		"""Train the model on the given 'trainData=(tInputs, tOutputs)' and check validation with 'validData(vInputs, vOutputs)' for 'epoch' amount of time.
		For both, the len of inputs and outputs should be equal, also, trainData and validData should be differentfor coherent result.
		- learningRate - (0.00 < learningRate <= 1.00 ) - determine the 'speed' of the learning process, lower the value, lower the model learn, but better it learn.
		- lossFunc - 
		- bathSize - (0 <= batchSize <= len(inputs)) - change the way the model use the data, if 0, it will take the whole tInputs len as value.
		- epoch - (0 < epoch < inf) - determine the amount of time the model will train on the given data.
		- verbose - (True/False), show the evolution epoch per epoch.
		- optimizer - 'GradientDescent'/'NesterovMomentum', training method choosing.
		- earlyStopping - (int > 0, float > 0.0), set a learning stop condition to avoid overfitting
		"""
		assert len(trainData[0]) == len(trainData[1]) and len(validData[0]) == len(validData[1]), "(Inputs, Outputs) len missmatch."
		assert trainData[0].shape and trainData[1].shape and validData[0].shape and validData[1].shape
		assert batchSize > 0 and learningRate > 0 and epoch > 0, "All integer value should be positive."

		#Loss Function setting -------------------------------------------------------
		match lossFunc:
			case 'categoricalCrossentropy':
				self._lossFunc = mlp_utils.categoricalCrossentropy
			case _:
				raise AssertionError(f'Invalid loss function: {lossFunc}')

		#Fill input/output layer -----------------------------------------------------
		self.__fillTrainIO(trainData[0], trainData[1])

		#LearningRate setting --------------------------------------------------------
		self._learningRate = learningRate

		#Training Param reset --------------------------------------------------------
		self._costPerEpoch = []
		self._accuracy = []

		#Validation Model creation ---------------------------------------------------
		vMod = _validation(self, validData[0], validData[1])

		#Routine setting -------------------------------------------------------------
		routine : routine_model
		match optimizer:
			case 'NesterovMomentum':
				routine = _NesMomentum(self, 0.9)
			case 'GradientDescent':
				routine = self
			case '':
				routine = self
			case _:
				raise AssertionError(f'Invalid training method: {optimizer}')

		#Early stopping setting ------------------------------------------------------
		apply_ES = False
		if earlyStopping != (0, 0.0):
			self._minEcart = earlyStopping[1]
			apply_ES = True

		if (batchSize > len(self._trainInputs)):
			batchSize = len(self._trainInputs)

		#Epoch Loop ------------------------------------------------------------------
		for i in range(epoch):

			#Training part -----------------------------------------------------------
			batchCost = []
			batchAcc = []
			samplePerBatch = []

			step = int(0)
			while step != -1: #loop over all dataset following the batchsize step
				step = self.__updateNodes(step, batchSize)
				samplePerBatch.append(self._sampleAmount)
				(co, acc) = routine.routine()
				batchCost.append(co)
				batchAcc.append(acc)

			self._costPerEpoch.append(mlp_utils.ponderate(batchCost, samplePerBatch))	#ponderate all batch cost to define the epoch cost
			self._accuracy.append(mlp_utils.ponderate(batchAcc, samplePerBatch))		#same for acc
			if (verbose is True):
				print(f"epoch {i + 1} / {epoch} | cost = {self._costPerEpoch[-1]:.4f} | accuracy = {self._accuracy[-1]:.4f}", end=' ')

			#Validation part ---------------------------------------------------------
			vMod.updateBias(self._layers)	#apply the bias modification on the validation layers
			vMod.fwdPropagation()			#feedforward

			self._validCost = self._lossFunc(vMod.validLayers[-1].activate(), vMod.expectedOutput)
			self._validcostPerEpoch.append(mlp_utils.batchCost(self._validCost, vMod.expectedOutput))
			self._validAccuracy.append(mlp_utils.calculateAccuracy(vMod.validLayers[-1].activate(), vMod.expectedOutput))
			if (verbose is True):
				print(f"| validCost = {self._validcostPerEpoch[-1]:.4f} | validAccuracy = {self._validAccuracy[-1]:.4f}")

			if (apply_ES is True):
				c = self._earlyStopping()
				if (c >= earlyStopping[0]):
					print("EarlyStop !")
					break

			#Data organization (shuffle) ---------------------------------------------
			shufInputs = self._layers[0].nodes
			shufOutput = self._expectedOutput
			(self._layers[0].nodes, self._expectedOutput) = mlp_utils.unison_shuffled_copies(shufInputs, shufOutput)	#will shuffle all training inputs and outputs by respect of their correspondig index
			self._layers[0]._activate = self._layers[0].nodes
		
		#-------------------------------------------------------------------------
		return self

	def useModel(self, testData: tuple[list, list]) -> tuple[float, float]:
		"""
		useModel will perform a prediction on the given set of data,
		it'll return a tuple containing the cost and the accuracy obtained.
		"""

		testInputs = testData[0]
		testOutputs = testData[1]
		assert len(testInputs[0]) == self._layers[0].size()
		assert len(testOutputs[0]) == self._layers[-1].size()
		vMod = _validation(self, testInputs, testOutputs)
		vMod.fwdPropagation()
		testCost = self._lossFunc(vMod.validLayers[-1].activate(), vMod.expectedOutput)
		cost = mlp_utils.batchCost(testCost, vMod.expectedOutput)
		acc = mlp_utils.calculateAccuracy(vMod.validLayers[-1].activate(), vMod.expectedOutput)
		return (cost, acc)

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