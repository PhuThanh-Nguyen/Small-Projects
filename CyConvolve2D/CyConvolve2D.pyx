# cython: boundscheck=False
# cython: wraparound=False
from numpy import zeros as npZeros, double as npDouble, asarray as AsNumpyArray

cdef double[:, :]getNeighborhood(double[:, ::1] arr, double[:, ::1] kernel, int i, int j):
	cdef:
		int kernel_rows, kernel_cols, pad_row, pad_col
	
	kernel_rows, kernel_cols = kernel.shape[:2]
	pad_row = 0 if kernel_rows % 2 == 0 else 1
	pad_col = 0 if kernel_cols % 2 == 0 else 1
	return arr[(i - kernel_rows//2) : (i + kernel_rows //2 + pad_row), (j - kernel_cols//2) : (j + kernel_cols//2 + pad_col)]

cdef double SumAndMultipleByElement(double[:, :] arr1, double[:, :] arr2):
	cdef:
		int N1 = arr1.shape[0], M1 = arr1.shape[1], N2 = arr2.shape[0], M2 = arr2.shape[1]
		int i, j
		double result = 0.0
	
	if (N1 != N2) or (M1 != M2):
		raise IndexError('Mismatch dimension')
	
	for i in range(N1):
		for j in range(M1):
			result += arr1[i, j] * arr2[i, j]
	return result

def convolve2D(double[:, ::1] arr, double[:, ::1] kernel):
	'''
		convolve2D(arr, kernel)
		Perform convolution 2D kernel to 2D array with zero-padding
		------------------------------
		Parameters:
			arr: 2D numpy.ndarray with dtype is float64
				2D array to convolve
			kernel: 2D numpy.ndarray with dtype is float64
				2D kernel to convolve with
		-----------------------------
		Returns: numpy.ndarray
			Result of convolution of input arrays
	'''
	cdef:
		int arr_rows, arr_cols, kernel_rows, kernel_cols, i, j
		double[:, ::1] pad_arr, result_arr
		double[:, :] neighbor
	
	arr_rows, arr_cols = arr.shape[:2]
	kernel_rows, kernel_cols = kernel.shape[:2]
	
	pad_arr = npZeros((arr_rows + kernel_rows * 2, arr_cols + kernel_cols * 2), dtype = npDouble)
	pad_arr[(kernel_rows + 1) : (kernel_rows + arr_rows + 1), (kernel_cols + 1) : (kernel_cols + arr_cols + 1)] = arr.copy()
	
	result_arr = pad_arr.copy()
	
	for i in range(kernel_rows + 1, kernel_rows + arr_rows + 1):
		for j in range(kernel_cols + 1, kernel_cols + arr_cols + 1):
			neighbor = getNeighborhood(pad_arr, kernel, i, j)
			result_arr[i, j] = SumAndMultipleByElement(neighbor, kernel)
	result_arr = result_arr[(kernel_rows + 1) : (kernel_rows + arr_rows + 1), (kernel_cols + 1) : (kernel_cols + arr_cols + 1)]
	return AsNumpyArray(result_arr)
