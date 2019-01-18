import numpy as np

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
	"""Parse a record to an image and perform data preprocessing.

	Args:
		record: An array of shape [3072,]. One row of the x_* matrix.
		training: A boolean. Determine whether it is in training mode.

	Returns:
		image: An array of shape [32, 32, 3].
	"""
	# Reshape from [depth * height * width] to [depth, height, width].
	# depth_major = tf.reshape(record, [3, 32, 32])
	depth_major = record.reshape((3, 32, 32))

	# Convert from [depth, height, width] to [height, width, depth]
	# image = tf.transpose(depth_major, [1, 2, 0])
	image = np.transpose(depth_major, [1, 2, 0])

	image = preprocess_image(image, training)
	# image = (image - np.mean(image)) / np.std(image)  ####
	return image


def preprocess_image(image, training):
	"""Preprocess a single image of shape [height, width, depth].

	Args:
		image: An array of shape [32, 32, 3].
		training: A boolean. Determine whether it is in training mode.

	Returns:
		image: An array of shape [32, 32, 3].
	"""
	if training:
		### YOUR CODE HERE
		# Resize the image to add four extra pixels on each side.
		# image = tf.image.resize_image_with_crop_or_pad(image, 32 + 8, 32 + 8)
		image = np.pad(image, ((4, 4), (4, 4), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
		# print(image.shape)
		### END CODE HERE

		### YOUR CODE HERE
		# Randomly crop a [32, 32] section of the image.
		# image = tf.random_crop(image, [32, 32, 3])
		# HINT: randomly generate the upper left point of the image
		up = np.random.randint(9)
		left = np.random.randint(9)
		image = image[up: up + 32, left: left + 32, :]
		# print([up, left])
		# print(image.shape)
		### END CODE HERE

		### YOUR CODE HERE
		# Randomly flip the image horizontally.
		# image = tf.image.random_flip_left_right(image)
		if np.random.random() > 0.5:
			image = np.flip(image, axis=1)
		### END CODE HERE

	### YOUR CODE HERE
	# Subtract off the mean and divide by the variance of the pixels.
	# image = tf.image.per_image_standardization(image)
	image = (image - np.mean(image)) / np.std(image)
	### END CODE HERE

	return image

if __name__ == '__main__':
	image = np.asarray([[[1]*32 for i in range(32)] for j in range(3)])
	image = image.transpose([1, 2, 0])
	preprocess_image(image, True)