import tensorflor as tf

class GazeTrackingModel(object):

	def __init__(self, isTraining):

		# Set the graph.
		self.graph = tf.Graph()

		with self.graph.as_default():

			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self.buildModel()

	def buildModel():

		self.X = tf.placeholder(tf.float32, shape=[-1,512,512,3])
		self.Y = tf.placeholder(tf.float32)

		with tf.variable_scope('FaceNetwork'):
			self.faceCnn0 = tf.contrib.layers.conv2d(self.X,
													 num_outputs = 256,
													 kernel_size = 15,
													 stride = 1,
													 padding='VALID')
			self.faceCnn1 = tf.contrib.layers.conv2d(self.faceCnn0,
													 num_outputs = 96,
													 kernel_size = 11,
													 stride = 1,
													 padding='VALID')
			self.faceCnn2 = tf.contrib.layers.conv2d(self.faceCnn1,
													 num_outputs = 256,
													 kernel_size = 5,
													 stride = 1,
													 padding='VALID')
			self.faceCnn3 = tf.contrib.layers.conv2d(self.faceCnn2,
													 num_outputs = 384,
													 kernel_size = 3,
													 stride = 1,
													 padding='VALID')
			self.faceCnn4 = tf.contrib.layers.conv2d(self.faceCnn3,
													 num_outputs = 64,
													 kernel_size = 1,
													 stride = 1,
													 padding='VALID')
			self.faceCnn4Flat = tf.reshape(self.faceCnn4, [cfg.batchSize, -1])
			self.faceDense1 = tf.layers.dense(self.faceCnn4_flat, units=128, activation=tf.nn.relu)
			self.faceDense2 = tf.layers.dense(self.faceDense1, units=64, activation=tf.nn.relu)


		with tf.variable_scope('RightEyeNetwork'):
			self.rightEyeCnn1 = tf.contrib.layers.conv2d(self.X,
													 num_outputs = 96,
													 kernel_size = 11,
													 stride = 1,
													 padding='VALID')
			self.rightEyeCnn2 = tf.contrib.layers.conv2d(self.rightEyeCnn1,
													 num_outputs = 256,
													 kernel_size = 5,
													 stride = 1,
													 padding='VALID')
			self.rightEyeCnn3 = tf.contrib.layers.conv2d(self.rightEyeCnn2,
													 num_outputs = 384,
													 kernel_size = 3,
													 stride = 1,
													 padding='VALID')
			self.rightEyeCnn4 = tf.contrib.layers.conv2d(self.rightEyeCnn3,
													 num_outputs = 64,
													 kernel_size = 1,
													 stride = 1,
													 padding='VALID')
			self.rightEyeCnn4Flat = tf.reshape(self.rightEyeCnn4, [cfg.batchSize, -1])
			self.rightEyeDense1 = tf.layers.dense(self.rightEyeCnn4_flat, units=128, activation=tf.nn.relu)
			self.rightEyeDense2 = tf.layers.dense(self.rightEyeDense1, units=64, activation=tf.nn.relu)

		with tf.variable_scope('LeftEyeNetwork'):
			self.leftEyeCnn1 = tf.contrib.layers.conv2d(self.X,
													 num_outputs = 96,
													 kernel_size = 11,
													 stride = 1,
													 padding='VALID')
			self.leftEyeCnn2 = tf.contrib.layers.conv2d(self.leftEyeCnn1,
													 num_outputs = 256,
													 kernel_size = 5,
													 stride = 1,
													 padding='VALID')
			self.leftEyeCnn3 = tf.contrib.layers.conv2d(self.leftEyeCnn2,
													 num_outputs = 384,
													 kernel_size = 3,
													 stride = 1,
													 padding='VALID')
			self.leftEyeCnn4 = tf.contrib.layers.conv2d(self.leftEyeCnn3,
													 num_outputs = 64,
													 kernel_size = 1,
													 stride = 1,
													 padding='VALID')
			self.leftEyeCnn4Flat = tf.reshape(self.leftEyeCnn4, [cfg.batchSize, -1])
			self.leftEyeDense1 = tf.layers.dense(self.leftEyeCnn4_flat, units=128, activation=tf.nn.relu)
			self.leftEyeDense2 = tf.layers.dense(self.leftEyeDense1, units=64, activation=tf.nn.relu)

		with tf.variable_scope('FaceBinaryOverlapNetwork'):
			self.binaryOverlapFlat = tf.reshape(self.X, [cfg.batchSize, -1])
			self.binaryOverlapDense1 = tf.layers.dense(self.binaryOverlapFlat, units=128, activation=tf.nn.relu)
			self.binaryOverlapDense2 = tf.layers.dense(self.binaryOverlapDense1, units=64, activation=tf.nn.relu)

		with tf.variable_scope('CombineEyes'):
			self.eyeDenseInput = tf.concat([self.rightEyeDense2, leftEyeDense2])
			self.eyesDense1 = tf.layers.dense(self.eyesDenseInput, units=128, activation=tf.nn.relu)

		with tf.variable_scope('CombineAll'):
			self.finalCombine = tf.concat([self.eyesDense1, self.faceDense2, self.binaryOverlapDense2])
			self.finalDense1 = tf.layers.dense(self.finalCombine, units=128, activation=tf.nn.relu)
			self.finalDense2 = tf.layers.dense(self.finalDense1, units=64, activation=tf.nn.relu)
			self.logits = tf.layers.dense(self.finalDense2, units=2, activation=None)

		with tf.variable_scope('Loss'):
			self.totalLoss = tf.constant(0.0)

		with tf.variable_scope('TrainOperation'):
			self.optimizer = tf.train.AdamOptimizer()
			self.trainOp = self.optimizer.minimize(self.totalLoss, global_step=self.globalStep)	

		with tf.variable_scope('Accuracy'):
			self.accuracy = tf.constant(1.0) - self.totalLoss