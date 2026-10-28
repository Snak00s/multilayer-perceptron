import numpy as np

class confMatrix:

	matrix: np.ndarray
	classList: np.ndarray

	def __init__(self, clas: int):
		self.clas = clas
		self.matrix = np.zeros((clas, clas), dtype=int)
		self.classList = np.identity(clas, dtype=int)
		return

	def clear(self):
		self.matrix = np.zeros((self.clas, self.clas), dtype=int)
		return

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
		return np.sum(self.matrix) - (self.TP_class(i) + self.FN_class(i) + self.FP_class(i))

	def TN(self):
		return np.sum([self.TN_class(i) for i in range(self.clas)])

	def __repr__(self):
		return f"{self.matrix}"