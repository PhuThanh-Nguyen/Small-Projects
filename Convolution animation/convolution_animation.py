import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

def getNeighborhood(img, kernel_shape, i, j):
	krows, kcols = kernel_shape
	return img[(i - krows//2):(i + krows//2 + 1), (j - kcols//2):(j + kcols//2 + 1)].copy() 

def convolution_timelapse(img, kernel):
	krows, kcols = kernel.shape[:2]
	result = np.ones(img.shape, dtype = img.dtype)
	result[0, 0] = 0.5
	
	padded_img = np.zeros((img.shape[0] + 2 * (krows//2), img.shape[1] + 2 * (kcols//2)), dtype = img.dtype)
	padded_img[krows//2: (krows//2 + img.shape[0]), kcols//2: (kcols//2 + img.shape[1])] = img.copy()
	
	for i in range(krows//2, krows//2 + img.shape[0]):
		for j in range(kcols//2, kcols//2 + img.shape[1]):
			neighbor = getNeighborhood(padded_img, (krows, kcols), i, j)
			result[i - krows//2, j - kcols//2] = np.sum(neighbor * kernel)
			yield result, neighbor

KERNEL = 1/273 * np.array(
	[
		[1, 4, 7, 4, 1],
		[4, 16, 26, 16, 4],
		[7, 26, 41, 26, 7],
		[4, 16, 26, 16, 4],
		[1, 4, 7, 4, 1]
	]
, dtype = np.float32)
WINDOW_SIZE = KERNEL.shape[0]

img = cv2.cvtColor(plt.imread('7.jpg'), cv2.COLOR_RGB2GRAY)

if ((img >= 0) & (img <= 1)).all():
	pass
else:
	img = img.astype(np.float32)/255.0
img = cv2.resize(img, (16, 16))
timelapse = convolution_timelapse(img, KERNEL)
ax_titles = ("Original image", "Snapshot of original image", "Kernel", "Output image")
i = 0
for img_conv, snapshot in timelapse:
	fig = plt.figure(figsize = (12, 8), constrained_layout = False)
	gs = fig.add_gridspec(4, 9)
	img_ax, snapshot_ax, kernel_ax, output_ax = (
		fig.add_subplot(gs[2:4, 0:4]),
		fig.add_subplot(gs[0:2, 2:4]),
		fig.add_subplot(gs[0:2, 5:7]),
		fig.add_subplot(gs[2:4, 5:9])
	)

	axes = (img_ax, snapshot_ax, kernel_ax, output_ax)

	for ax, title in zip(axes, ax_titles):
		ax.set(xticks = [], yticks = [], title = title)

	img_ax.imshow(img, cmap = plt.cm.gray)
	kernel_ax.imshow(KERNEL, cmap = plt.cm.gray)
	snapshot_ax.imshow(snapshot, cmap = plt.cm.gray)
	output_ax.imshow(img_conv, cmap = plt.cm.gray)
	fig.savefig(f"images/t_{i}.jpg")
	i += 1
	plt.close("all")
