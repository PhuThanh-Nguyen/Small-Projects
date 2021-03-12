import numpy as np
from numpy.linalg import eigh
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

def getCovariance(X):
	'''
		Calculate covariance matrix, given data matrix where each row is an observation.
		Parameter:
			X: np.ndarray
				Data matrix
		Returns: np.ndarray
			Covariance matrix
	'''
	return np.cov(X, rowvar = False, ddof = 1)

def getEigen(Sigma):
	'''
		Get a list of tuples (eigenvalue, eigenvectors), given covariance matrix
		Parameter:
			Sigma: np.ndarray
				Covariance matrix (symmetric matrix)
		Returns: list
	'''
	eigenvals, eigenvecs = eigh(Sigma)
	
	eigenpairs = [(abs(eigenval), eigenvec) for (eigenval, eigenvec) in zip(eigenvals, eigenvecs.T)]
	
	eigenpairs = sorted(eigenpairs, key = lambda pair: pair[0], reverse = True)
	
	return eigenpairs

def PCA(X, k = None):
	'''
		Perform PCA on data matrix.
		Parameter:
			X: np.ndarray
				Data matrix
			k: int, default: None
				If k is None, then returns all eigenpairs
				Otherwise, returns the first k eigenpairs
	'''
	Sigma = getCovariance(X)
	
	eigenpairs = getEigen(Sigma)
	
	if k is None:
		return eigenpairs
	else:
		return eigenpairs[:k]

X, y = load_digits(return_X_y = True)

NUMBER_PCA = 10

fig, axes = plt.subplots(nrows = NUMBER_PCA, ncols = 10, figsize = (14, 12))

for label in range(10):
	subset = X[y == label]
	eigenpairs = PCA(subset, k = NUMBER_PCA)
	eigenvecs = [eigenvec for (_, eigenvec) in eigenpairs]
	
	for i, eigenvec in enumerate(eigenvecs):
		axes[i][label].set(xticks = [], yticks = [])
		if i == 0:
			axes[i][label].set(title = f'{label}')
		axes[i][label].imshow(eigenvec.reshape((8,8)), cmap = plt.cm.gray)
fig.savefig(f'PCA MNIST with {NUMBER_PCA} principle components.jpg')
