import numpy as np

def backtrackingLineSearch(func, gradient, x, direction, alpha = 0.5, beta = 0.5):
	'''
		Perform Backtracking line search
		
		Parameters:
			func: Convex function
			gradient: Gradient function of convex function
			x: np.ndarray or floats, must in domain of convex function
			direction: np.ndarray or floats, must have the same type as x
			alpha, beta: floats
				Controlling rates, alpha needs to be in range (0, 0.5) and beta needs to be in range (0, 1),
				default: Alpha = 0.5 and Beta = 0.5
		
		Returns:
			t: float
				Step size to update x
	'''
	
	t, grad, f_x = 1, gradient(x), func(x)
	const = alpha*grad.T.dot(direction)
	
	while func(x + t * direction) > (f_x + t*const):
		t = beta*t
	
	return t

class EqualityConstrained:
	
	def __init__(self, func, gradient, hessian, A, b):
	
		'''
			Solving equality constrained minimization problem which has a form:
					minimize f(x)
					subject to Ax = b
			where x in R^n, A is a m x n matrix, b is a m column vector and f is a convex function
			
			Parameters:
				func, gradient, hessian: Correspond to f in the problem with its gradient and hessian functions
				A, b: np.ndarray
		'''
		
		self.f, self.grad, self.hess = func, gradient, hessian
		self.A, self.b = A, b
		self.shape = A.shape[1]
	
	def getKKT(self, x):
	
		'''
			Get KKT matrix function at given point x.
			
			KKT matrix of the problem has the block form:
				[[Hessian(x)	A^T]
				 [A				0  ]]
			
			Parameters:
				x: np.ndarray or float, needs to be in domain of f
			
			Returns:
				mat: np.ndarray of shape (n + m, n + m)
					KKT Matrix at point x
		'''
	
		m = self.A.shape[0]
		
		upper = np.hstack((self.hess(x), self.A.T))
		lower = np.hstack((self.A, np.zeros((m, m))))
		mat = np.vstack((upper, lower))
		
		return mat
	
	def getNewtonStep(self, x):
		
		'''
			Compute Newton step and Newton decrement
			
			Newton step delta_x_nt is characterized by:
				[[Hessian(x)	A^T] [ delta_x_nt	= [ -Gradient(x)
				 [A				0  ]]	w]					0]
			
			Newton decrement is:
				lambda(x) = [(delta_x_nt^T)(Hessian(x))(delta_x_nt^T)]^(1/2)
			
			Parameters:
				x: np.ndarray or float, needs to be in domain of f
			
			Returns:
				newton_step, newton_decrement: tuple
					Newton step and Newton decrement at point x
					Newton step has shape (n, 1) and Newton decrement is a floating number
		'''
		
		m, n = self.A.shape
		mat = self.getKKT(x)
		b = np.vstack(
			(
				-self.grad(x), 
				np.zeros(m).reshape(m, 1)
			)
		)
		
		vec = np.linalg.solve(mat, b)
		
		newton_step = vec[:n, 0].reshape((n, 1))
		newton_decrement = np.sqrt(newton_step.T.dot(self.hess(x)).dot(newton_step))
		
		return newton_step, newton_decrement
	
	def NewtonMethod(self, initial, tolerance = 1e-3, max_iter = 1000, alpha = 0.5, beta = 0.5):
		'''
			Perform Newton's method
			
			Parameters:
				initial: np.ndarray
					Initial starting point
				tolerance: float, positive, default 1e-3
					Quits if Newton decrement square is less than 2*tolerance
				max_iter: integer, positive, default: 1000
					Maximum number of iterations
				alpha, beta: float, default: alpha = 0.5 and beta = 0.5
					alpha needs to be in range (0, 0.5) and beta needs to be in range (0, 1)
					Controlling rates in backtracking line search
			Returns:
				x, decrement: tuple
					x: is a solution to the equality constrained convex problem, x has the shape of (n, 1)
					decrement: Newton decrement at x 
		'''
		assert self.A.shape[1] == initial.shape[0]
		x = initial
		for _ in range(max_iter):
			newton_step, decrement = self.getNewtonStep(x)
			if decrement**2/2 < tolerance:
				return x, decrement
			step = backtrackingLineSearch(self.f, self.grad, x, newton_step, alpha, beta)
			x = x + step * newton_step
		
		print('Warning: Maximum iteration has reached. Result may not converges to optimal solution yet, you may need to increase max_iter')
		return x, decrement
