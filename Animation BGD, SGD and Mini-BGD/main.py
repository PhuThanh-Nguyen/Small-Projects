from Optimizers import BatchGradientDescent, MiniBatchGradientDescent, StochasticGradientDescent
import numpy as np
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation

def objective_func(theta, X, y):
	sample_size, dimensions = X.shape[:2]
	residuals = X @ theta - y
	return 1/(2 * sample_size) * residuals.T @ residuals

def gradient(theta, X, y):
	sample_size, dimensions = X.shape[:2]
	return 1/sample_size * X.T @ (X @ theta - y)

X, y = make_regression(n_samples = 100, n_features = 1, n_informative = 1, n_targets = 1, bias = 10, noise = 10, random_state = 0)

X = np.hstack((np.ones((X.shape[0], 1)), X))
y = y.reshape(-1, 1)
optimum = np.linalg.inv(X.T @ X) @ X.T @ y
init_sol = np.array([10.5, 42.5]).reshape(-1, 1)

theta_1 = np.arange(8, 11, 0.025)
theta_2 = np.arange(41, 44, 0.025)
theta_1, theta_2 = np.meshgrid(theta_1, theta_2)
thetas = np.vstack((theta_1.reshape(1, -1), theta_2.reshape(1, -1)))
objectives = np.sum((X @ thetas - y)**2, axis = 0).reshape(theta_1.shape)

kwargs_init = {
	"learning_rate": 5e-3,
	"gradient_func": gradient
}

kwargs_fit = {
	"X": X,
	"y": y,
	"init_sol": init_sol,
	"random_state": 0
}

trajectories = {
	"BGD": [],
	"MBGD": [],
	"SGD": []
}

BGD = BatchGradientDescent(**kwargs_init, nb_epoch = 200)
MBGD = MiniBatchGradientDescent(**kwargs_init, batch_size = 5, nb_epoch = 10)
SGD = StochasticGradientDescent(**kwargs_init, nb_epoch = 2)
for theta_BGD, theta_MBGD, theta_SGD in zip(
	BGD.fit(**kwargs_fit), 
	MBGD.fit(**kwargs_fit), 
	SGD.fit(**kwargs_fit)):
	
	trajectories["BGD"].append(theta_BGD.copy())
	trajectories["MBGD"].append(theta_MBGD.copy())
	trajectories["SGD"].append(theta_SGD.copy())

for key in trajectories.keys():
	trajectories[key] = np.hstack(trajectories[key]).T

fig = plt.figure(figsize = (12, 10), constrained_layout=True)
gs = fig.add_gridspec(4, 4)
axes = [fig.add_subplot(gs[:2, :2]), fig.add_subplot(gs[:2, 2:]), fig.add_subplot(gs[2:, 1:3])]
i = 1
ax_titles = ["Batch Gradient Descent", "Mini-Batch Gradient Descent", "Stochastic Gradient Descent"]
for index, ax in enumerate(axes):
	ax.contour(theta_1, theta_2, objectives, levels = 50)
	ax.scatter(optimum[0], optimum[1], c = 'r')
	ax.set(xticks = [], yticks = [], title = ax_titles[index])

def update_animation(frame):
	global axes, i, trajectories
	for index, (ax, (key, trajectory)) in enumerate(zip(axes, trajectories.items())):
		ax.cla()
		ax.contour(theta_1, theta_2, objectives, levels = 50)
		ax.scatter(optimum[0], optimum[1], c = 'r')
		ax.set(xticks = [], yticks = [], title = ax_titles[index], xlim = ax.get_xlim(), ylim = ax.get_ylim())
		ax.plot(trajectory[:i, 0], trajectory[:i, 1], marker = 'o', markersize = 1, color = 'blue')
	if i < 200:
		i += 1
	else:
		i = 1
ani = FuncAnimation(fig, update_animation, interval = 1, frames = 200)
plt.close(ani._fig)
ani.save('animation.gif', writer='ffmpeg')
