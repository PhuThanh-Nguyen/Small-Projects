import numpy as np

# Abstract class
class Optimizer:
	def __init__(self, nb_epoch, learning_rate, gradient_func):
		self.nb_epoch, self.rate = nb_epoch, learning_rate
		self.grad = gradient_func
	def fit(self, X, y, init_sol = None, random_state = 0):
		raise NotImplementedError

class BatchGradientDescent(Optimizer):
	def __init__(self, nb_epoch, learning_rate, gradient_func):
		super(BatchGradientDescent, self).__init__(nb_epoch, learning_rate, gradient_func)
	
	def fit(self, X, y, init_sol = None, random_state = 0):
		y = y.reshape(-1, 1)
		rnd = np.random.RandomState(random_state)
		if init_sol is None:
			self.theta = rnd.normal(loc = 0, scale = 1e-2, size = (X.shape[1], 1))
		else:
			self.theta = init_sol.copy()
		
		yield self.theta
		for _ in range(self.nb_epoch):	
			self.theta -= self.rate * self.grad(self.theta, X, y)
			yield self.theta
class MiniBatchGradientDescent(Optimizer):
	def __init__(self, nb_epoch, learning_rate, gradient_func, batch_size = 16):
		super(MiniBatchGradientDescent, self).__init__(nb_epoch, learning_rate, gradient_func)
		self.batch_size = batch_size
	
	def fit(self, X, y, init_sol = None, random_state = 0):
		X = X.copy()
		y = y.reshape(-1, 1)
		sample_size, dimension = X.shape[:2]
		rnd = np.random.RandomState(random_state)
		if init_sol is None:
			self.theta = rnd.normal(loc = 0, scale = 1e-2, size = (X.shape[1], 1))
		else:
			self.theta = init_sol.copy()
		
		yield self.theta
		for _ in range(self.nb_epoch):
			data = np.hstack((X, y))
			# Shuffle inplace
			rnd.shuffle(data)
			# Get back X, y after shuffle
			X, y = data[:, :dimension], data[:, dimension:]
			
			for i in range(0, sample_size, self.batch_size):
				X_batch, y_batch = X[i:(i + self.batch_size)], y[i:(i + self.batch_size)].reshape(-1, 1)
				self.theta -= self.rate * self.grad(self.theta, X_batch, y_batch)
				yield self.theta

class StochasticGradientDescent(MiniBatchGradientDescent):
	def __init__(self, nb_epoch, learning_rate, gradient_func):
		super(StochasticGradientDescent, self).__init__(nb_epoch, learning_rate, gradient_func, batch_size = 1)
