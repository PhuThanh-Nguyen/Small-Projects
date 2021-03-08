import numpy as np
from matplotlib import pyplot as plt
from robustReg import RobustRegression
from sklearn.linear_model import LinearRegression

rnd = np.random.RandomState(0)

x = np.linspace(0, 2, num = 50)
y = 2*x + 1

noise = rnd.normal(loc = 0, scale = 0.1, size = y.shape)
y += noise

y[3] += 10
y[10] -= 10
y[25] -= 10
y[37] += 10

col_x = x.reshape((50, 1))

X = np.hstack(
	(
		col_x,
		col_x**2,
		col_x**3,
	)
)

model = RobustRegression(X, y)
reg = LinearRegression().fit(X, y)

theta = model.findSolution()

fig, ax = plt.subplots(figsize = (12, 10))

ax.scatter(x, y)
ax.scatter([x[3], x[10], x[25], x[37]], [y[3], y[10], y[25], y[37]], color = 'red', label = 'Outliers')
ax.plot(x, X.dot(theta[1:]) + theta[0], color = 'red', label = 'Huber Regression')
ax.plot(x, reg.predict(X), color = 'blue', label = 'Ordinary Regression')

ax.legend()
fig.savefig('Huber vs Ordinary regression.jpg')
