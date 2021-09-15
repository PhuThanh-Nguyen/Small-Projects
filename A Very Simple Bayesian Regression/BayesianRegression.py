import numpy as np, matplotlib.pyplot as plt

def getBatches(X, y, batch_size):
	sample_size, dimension = X.shape[:2]
	for i in range(0, sample_size, batch_size):
		yield X[i:(i+batch_size), :].reshape(-1, dimension), y[i:(i+batch_size)].reshape(-1, 1)

class BayesianRegression:
	def __init__(self, prior_mean, prior_cov):
		'''
		Initialize Normal prior distribution.
		--------------------------------------------
		Parameters:
			prior_mean: numpy.ndarray of shape (p, 1)
				Mean of Multivariate Normal prior distribution
			prior_cov: numpy.ndarray of shape (p, p)
				Covariance matrix of Multivariate Normal prior distribution
		--------------------------------------------
		Example:
		>>> mean = np.zeros(3).reshape(-1, 1) # Zeros vector
		>>> cov = np.eye(3) # 3x3 identity matrix
		>>> regr = BayesianRegression(mean, cov)
   		--------------------------------------------
		'''
		self.mean, self.cov = prior_mean.copy(), prior_cov.copy()
	def fit(self, X, y, max_iter = 100, batch_size = None, shuffle = True, random_state = 0):
		'''
		Fit Bayesian Regression to dataset
		--------------------------------------------
		Paramters:
			X: np.ndarray of shape (n, p)
				Training dataset with n samples, each sample is a vector in p-dimension
			y: np.ndarray of shape (n,) or (n, 1)
				Training output with n outputs
			max_iter: int, default 100
				Number of iterations
			batch_size: int, default None
				Number of samples that prior distribution use to calculate posterior at each iteration
				If batch_size is None, all samples in X will be used and shuffle parameter will be ignored
			shuffle: boolean, default True
				If True, at each iteration, dataset will be shuffled
				If batch_size is None then shuffle will be ignored (reassign to be False)
			random_state: int, default 0
		'''
		rnd = np.random.RandomState(random_state)
		y = y.reshape(-1, 1)
		
		if batch_size is None:
			batch_size = X.shape[0]
			shuffle = False
		
		for _ in range(max_iter):
			if shuffle:
				temp = np.hstack((X, y))
				rnd.shuffle(temp)
				X, y = temp[:, :-1].copy(), temp[:, -1].reshape(-1, 1).copy()
				del temp
			for batch_X, batch_y in getBatches(X, y, batch_size):
				# Calculate posterior statistics and then treat it like prior statistics for the next iteration
				precision = np.linalg.inv(self.cov)
				next_cov = np.linalg.inv(precision + batch_X.T @ batch_X)
				self.mean = (
					next_cov @ (
						precision @ self.mean + batch_X.T @ batch_y
					)
				)
				self.cov = next_cov.copy()
				del precision, next_cov
	def predict(self, X):
		'''
		Predict output from the new dataset
		--------------------------------
		Parameters:
			X: np.ndarray of shape (m, p)
		--------------------------------
		Returns: Tuple of two np.ndarray vectors of shape (m, 1)
			First element of tuple is mean calculated at each point
			Second element of tuple is variance calculated at each point
		'''
		if len(X.shape) == 1:
			X = X.reshape(1, -1)
		return X @ self.mean, np.diag(X @ self.cov @ X.T).reshape(-1, 1)
