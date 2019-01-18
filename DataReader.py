import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""


def load_data(data_dir):
    """Load the CIFAR-10 dataset.

	Args:
		data_dir: A string. The directory where data batches
			are stored.

	Returns:
		x_train: An numpy array of shape [50000, 3072].
			(dtype=np.float32)
		y_train: An numpy array of shape [50000,].
			(dtype=np.int32)
		x_test: An numpy array of shape [10000, 3072].
			(dtype=np.float32)
		y_test: An numpy array of shape [10000,].
			(dtype=np.int32)
	"""

    ### YOUR CODE HERE
    for i in range(1, 6):
        with open(data_dir + '/data_batch_' + str(i), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            if i > 1:
                x_train = np.concatenate((x_train, dict[b'data']), axis=0)
                y_train = np.concatenate((y_train, np.asarray(dict[b'labels'])), axis=0)
            else:
                x_train = dict[b'data']
                y_train = np.asarray(dict[b'labels'])

    with open(data_dir + '/test_batch', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        x_test = dict[b'data']
        y_test = np.asarray(dict[b'labels'])

    ### END CODE HERE
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    return x_train, y_train, x_test, y_test


def train_valid_split(x_train, y_train, split_index=45000):
    """Split the original training data into a new training dataset
	and a validation dataset.

	Args:
		x_train: An array of shape [50000, 3072].
		y_train: An array of shape [50000,].
		split_index: An integer.

	Returns:
		x_train_new: An array of shape [split_index, 3072].
		y_train_new: An array of shape [split_index,].
		x_valid: An array of shape [50000-split_index, 3072].
		y_valid: An array of shape [50000-split_index,].
	"""
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data('.')
    train_valid_split(x_train, y_train)
