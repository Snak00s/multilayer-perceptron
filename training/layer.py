import numpy as np
from dataclasses import dataclass

@dataclass
class layer:

	_prevLayer: layer
	_nextLayer: layer
	_delta: np.ndarray
	activFunc: function

	def __init__(self):
		self._idx = 0
		self._size = 0
		self.nodes = np.array([], dtype=np.float64)
		self.bias = np.array([], dtype=np.float64)
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
		obj.nodes = np.array([0 for _ in range(size)], dtype=np.float64)
		obj.bias = np.array([1 for _ in range(size)], dtype=np.float64)
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