from cvx import EqualityConstrained
import numpy as np
from math import copysign

def huber(x, M):
	'''
		Huber penalty function:
					  |[ x^2, if |x| <= M
			phi(x) =  |
					  |[ M(2|x| - M), otherwise 
	'''
	if abs(x) <= M:
		return x**2
	return M*(2*abs(x) - M)

def derivative_huber(x, M):
	'''
		First derivative of huber penalty function:
						|[ 2x, if |x| <= M
			phi'(x) = 	|
						|[ -2M, if x < -M
						|
						|[ 2M, if x > M
	'''
	if abs(x) <= M:
		return 2*x
	return copysign(2*M, x)

def hess_huber(x, M):
	'''
		Second derivative of huber penalty function:
					    |[ 2, if |x| <= M
			phi''(x) =  |
					    |[ 0, otherwise
	'''
	if abs(x) <= M:
		return 2
	return 0

class RobustRegression:
	
	def __init__(self, t, y, tol = 1):
	
		'''
			Perform simple robust linear regression
			Parameters:
			
				t: np.ndarray
					Input matrix, each row is an observation
					Otherwise, if t is a vector then it must be a row vector
					
				y: np.ndarray
					Output row vector
				
				tol: float, positive
					Parameter M in huber penalty function
		'''
		
		if len(t.shape) == 2:
			n, param = t.shape[:2]
		else:
			n, param = t.shape[0], 1
			t = t.reshape(n, 1)
		
		param += 1 # Count total of parameters, including theta_0
		
		self.param = param
		self.t, self.y = t, y.reshape(n, 1)
		
		# Convert original problem into equality constrained convex optimization problem (See PDF version)
		
		temp_huber = np.vectorize(lambda x: huber(x, tol))
		temp_deriv = np.vectorize(lambda x: derivative_huber(x, tol))
		temp_hess = np.vectorize(lambda x: hess_huber(x, tol))
		
		self.func = lambda x: np.sum(temp_huber(x.flatten()[:n]))
		self.grad = lambda x: np.vstack(
			(
				temp_deriv(x[:n]).reshape(n, 1),
				np.zeros(self.param).reshape(self.param, 1)
			)
		)
		
		
		self.hess = lambda x: np.vstack(
			(
				np.hstack(
					(
						np.diag(temp_hess(x.flatten()[:n])),
						np.zeros((n, self.param))
					)
				),
				np.hstack(
					(
						np.zeros((self.param, n)),
						np.zeros((self.param, self.param))	
					)
				)
			)
		)
		
		self.A = np.hstack(
			(
				-np.eye(n),
				np.ones(n).reshape(n, 1),
				self.t
			)
		)
		
		# Feasible initial starting point (See PDF)
		self.initial = np.vstack(
			(
				1 - self.y,
				1,
				np.zeros((self.param - 1, 1))
			)
		)
		
	def findSolution(self, tolerance = 1e-3, max_iter = 1000, alpha = 0.5, beta = 0.5):
		
		'''
			Find solution of the problem
			
			Parameters:
				tolerance: float, positive, default 1e-3
					Quits if Newton decrement square is less than 2*tolerance
				max_iter: integer, positive, default: 1000
					Maximum number of iterations
				alpha, beta: float, default: alpha = 0.5 and beta = 0.5
					alpha needs to be in range (0, 0.5) and beta needs to be in range (0, 1)
					Controlling rates in backtracking line search
			Returns:
				solution: np.ndarray
					A solution to robust linear regressiob (Huber penalty) given t and y
				
		'''
		problem = EqualityConstrained(self.func, self.grad, self.hess, self.A, self.y)
		solution, decrement = problem.NewtonMethod(self.initial, tolerance, max_iter, alpha, beta)
		
		# This means the solution might not be the optimal one.
		# Don't worry when this happens, cvx will have a warning for user beforehand
		if decrement**2/2 >= tolerance:
			print(f'>> Newton decrement of solution: {decrement**2/2}')
		return solution[-self.param:]
