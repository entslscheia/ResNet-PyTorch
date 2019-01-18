import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Network import ResNet
from ImageUtils import parse_record

"""This script defines the training, validation and testing process.
"""

class Cifar(object):

	def __init__(self, sess, conf):
		self.sess = sess
		self.conf = conf

	def setup(self, training):
		print('---Setup input interfaces...')
		self.inputs = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
		self.labels = tf.placeholder(tf.int32)
		# Note: this placeholder allows us to set the learning rate for each epoch
		self.learning_rate = tf.placeholder(tf.float32)

		print('---Setup the network...')
		network = ResNet(self.conf.resnet_version, self.conf.resnet_size,
					self.conf.num_classes, self.conf.first_num_filters)

		if training:
			print('---Setup training components...')
			# compute logits
			self.logits = network(self.inputs, True)
			# self.logits2 = network(self.inputs, False)

			# predictions for validation
			self.preds = tf.argmax(self.logits, axis=-1)
			# self.preds2 = tf.argmax(self.logits2, axis=-1)

			# weight decay
			l2_loss = self.conf.weight_decay * tf.add_n(
						[tf.nn.l2_loss(v) for v in tf.trainable_variables()
						if 'kernel' in v.name])

			### YOUR CODE HERE
			# cross entropy
			cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.logits)
			# final loss function
			self.losses = tf.add(l2_loss, cross_entropy_loss)
			### END CODE HERE

			# momentum optimizer with momentum=0.9
			optimizer = tf.train.MomentumOptimizer(
							learning_rate=self.learning_rate, momentum=0.9)

			### YOUR CODE HERE
			# train_op
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				self.train_op = optimizer.minimize(self.losses)

			### END CODE HERE
			# for var in tf.trainable_variables():
			# for var in tf.global_variables():
			# 	print(var.name)

			print('---Setup the Saver for saving models...')
			self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=0)



		else:
			print('---Setup testing components...')
			# compute predictions
			self.logits = network(self.inputs, False)
			self.preds = tf.argmax(self.logits, axis=-1)

			print('---Setup the Saver for loading models...')
			self.loader = tf.train.Saver(var_list=tf.global_variables())


	def train(self, x_train, y_train, max_epoch):
		print('###Train###')

		self.setup(True)
		self.sess.run(tf.global_variables_initializer())

		# Determine how many batches in an epoch
		num_samples = x_train.shape[0]
		num_batches = int(num_samples / self.conf.batch_size)
		print(y_train)
		print('---Run...')
		for epoch in range(1, max_epoch+1):

			start_time = time.time()
			# Shuffle
			shuffle_index = np.random.permutation(num_samples)
			curr_x_train = x_train[shuffle_index]
			curr_y_train = y_train[shuffle_index]

			### YOUR CODE HERE
			# Set the learning rate for this epoch
			# Usage example: divide the initial learning rate by 10 after several epochs
			learning_rate = 0.1

			### END CODE HERE

			loss_value = []
			preds = []
			# preds2 = []
			logits = []
			# logits2 = []
			for i in range(num_batches):
				### YOUR CODE HERE
				# Construct the current batch.
				# Don't forget to use "parse_record" to perform data preprocessing.
				if epoch % 10 == 0:
					learning_rate /= 10
				x_batch = curr_x_train[i * self.conf.batch_size: min((i + 1) * self.conf.batch_size, num_samples)]
				x_batch = list(map(lambda x: parse_record(x, True), x_batch))
				# print('np_mean: ', np.array(x_batch).mean(axis=(0, 1, 2)))
				y_batch = curr_y_train[i * self.conf.batch_size: min((i + 1) * self.conf.batch_size, num_samples)]

				### END CODE HERE

				# Run
				feed_dict = {self.inputs: x_batch,
							self.labels: y_batch,
							self.learning_rate: learning_rate}
				loss, _ = self.sess.run(
							[self.losses, self.train_op], feed_dict=feed_dict)
				###
				preds.append(self.sess.run(self.preds, feed_dict=feed_dict))
				# preds2.append(self.sess.run(self.preds2, feed_dict=feed_dict))
				logits.append(self.sess.run(self.logits, feed_dict=feed_dict))
				# logits2.append(self.sess.run(self.logits2, feed_dict=feed_dict))
				###
				print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss),
						end='\r', flush=True)

			duration = time.time() - start_time
			print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(
						epoch, loss, duration))
			###
			preds = np.array(preds).reshape(curr_y_train.shape)
			# preds2 = np.array(preds2).reshape(curr_y_train.shape)
			logits = np.array(logits)
			# logits2 = np.array(logits2)
			###
			print('Train accuracy: {:.4f}, {:d}/{:d}'.format(np.sum(preds == curr_y_train)/curr_y_train.shape[0],
															 np.sum(preds == curr_y_train), curr_y_train.shape[0]))
			# print('Train accuracy2: {:.4f}, {:d}/{:d}'.format(np.sum(preds2 == curr_y_train) / curr_y_train.shape[0],
			#												 np.sum(preds2 == curr_y_train), curr_y_train.shape[0]))
			# print('preds2: ', preds2)

			if epoch % self.conf.save_interval == 0:
				self.save(self.saver, epoch)
				# print('logits: ', logits[0][0])
				# print('logits2: ', logits2[0][0])
				# print('stack_layer_16/block_0/shortcut_v1/batch_normalization/moving_mean:0', self.sess.run('stack_layer_16/block_0/shortcut_v1/batch_normalization/moving_mean:0'))
				# print('stack_layer_16/block_0/shortcut_v1/batch_normalization/moving_variance:0',
				# 	  self.sess.run('stack_layer_16/block_0/shortcut_v1/batch_normalization/moving_variance:0'))
				# print('stack_layer_32/block_0/block_v1/batch_norm_relu/batch_normalization/moving_mean:0',
				# 	  self.sess.run('stack_layer_32/block_0/block_v1/batch_norm_relu/batch_normalization/moving_mean:0'))
                #
				# print('start_layer/batch_normalization/moving_mean:0', self.sess.run('start_layer/batch_normalization/moving_mean:0'))
				# print('start_layer/batch_normalization/moving_variance:0',
				# 	  self.sess.run('start_layer/batch_normalization/moving_variance:0'))

	def test_or_validate(self, x, y, checkpoint_num_list):
		print('###Test or Validation###')

		self.setup(False)
		self.sess.run(tf.global_variables_initializer())

		# load checkpoint
		for checkpoint_num in checkpoint_num_list:
			checkpointfile = self.conf.modeldir+'/model.ckpt-'+str(checkpoint_num)
			self.load(self.loader, checkpointfile)
			preds = []
			logits = []
			for i in tqdm(range(x.shape[0])):
				### YOUR CODE HERE
				feed_dict = {self.inputs: [parse_record(np.array(x[i]), False)], self.labels: y[i]}
				preds.append(self.sess.run(self.preds, feed_dict=feed_dict))
				logits.append(self.sess.run(self.logits, feed_dict=feed_dict))
				### END CODE HERE

			preds = np.array(preds).reshape(y.shape)
			# print('preds: ', preds[:50])
			# print('logits: ', logits[0])
			# print('y: ', y[:50])
			print('Test accuracy: {:.4f}'.format(np.sum(preds==y)/y.shape[0]))
			# print('stack_layer_32/block_0/block_v1/batch_norm_relu/batch_normalization/moving_mean:0',
			# 	  self.sess.run('stack_layer_32/block_0/block_v1/batch_norm_relu/batch_normalization/moving_mean:0'))
			# print('start_layer/batch_normalization/moving_mean:0',
			# 	  self.sess.run('start_layer/batch_normalization/moving_mean:0'))
			# print('start_layer/batch_normalization/moving_variance:0',
			# 	  self.sess.run('start_layer/batch_normalization/moving_variance:0'))

	def save(self, saver, step):
		'''Save weights.
		'''
		model_name = 'model.ckpt'
		checkpoint_path = os.path.join(self.conf.modeldir, model_name)
		if not os.path.exists(self.conf.modeldir):
			os.makedirs(self.conf.modeldir)
		saver.save(self.sess, checkpoint_path, global_step=step)
		print('The checkpoint has been created.')


	def load(self, loader, filename):
		'''Load trained weights.
		''' 
		loader.restore(self.sess, filename)
		print("Restored model parameters from {}".format(filename))

