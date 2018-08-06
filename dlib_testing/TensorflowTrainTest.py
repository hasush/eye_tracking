import os

import numpy as np
import tensorflow as tf

class TensorflowTrainTest(object):

	def __init__(self, model, logdir, dataXPath, dataYPath, batchSize, numFilesLoad, saveModelSecs=0):

		# Generate supervisor.
		self.sv = tf.train.Supervisor(graph=model.graph, logdir=logdir, save_model_secs=saveModelSecs)

		# Output tensorflow log information.
		self.logdir = logdir.

		# Path to the X and Y data.
		self.dataXPath = dataXPath
		self.dataYPath = dataYPath

		# The number of instances in a batch.
		self.batchSize = batchSize 

		# The number of "dataX{Y}_batch" files to load at once. 
		self.numFilesLoad = numFilesLoad 

		# Tensorflow configuration.
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True

		# Determine the number of batch files.
		self.determineNumberOfBatchFiles()

						
	def loadBatchData(self, batchNum):
		pass


	def load_mnist():
	    fd = open('train-images-idx3-ubyte')
	    loaded = np.fromfile(file=fd, dtype=np.uint8)
	    trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

	    fd = open('train-labels-idx1-ubyte')
	    loaded = np.fromfile(file=fd, dtype=np.uint8)
	    trainY = loaded[8:].reshape((60000)).astype(np.int32)

	    trX = trainX[:55000] / 255.
	    trY = trainY[:55000]

	    valX = trainX[55000:, ] / 255.
	    valY = trainY[55000:]

	    num_tr_batch = 55000 // batch_size
	    num_val_batch = 5000 // batch_size

	    return trX, trY, num_tr_batch, valX, valY, num_val_batch

	def determineNumberOfBatchFiles(self):

		# List contents of directory.		
		dataXPathContents = os.listdir(self.dataXPath)
		dataYPathContents = os.listdir(self.dataYPath)
		
		# Initalize the number of files to zero.
		numBatchFilesX = 0
		numBatchFilesY = 0

		# Loop over file in directory and see if it has the needed substring.
		for file in dataXPathContents:
			if 'dataX_batch' in file:
				numBatchFilesX +=1
		for file in dataYPathContents:
			if 'dataY_batch' in file:
				numBatchFilesY +=1

		# Make sure that the features and labels are equal.
		assert numBatchFilesX == numBatchFilesY

		# Set the member variable.
		self.numBatchFiles = numBatchFilesX

	def train(self):

		# Figure out how many batches of data will be utilized.

		# Enter the session.
		with self.sv.managed_session(config=self.config) as self.sess:

			# Loop over the epochs of the data.
			for epoch in range(self.numEpochs):

				# Check to see if any errors have been thrown.
				if sv.should_stop():
					print("Stopping the supervisor")
					break


				for step in range(num_tr_batch):
					start = step * batch_size
					end = start + batch

					self.sess.run(model.trainOp, )

	def test(self):
		pass

def get_batch_data(batch_size):
	trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist()					

def main():

	# Load mnist data.
	trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist()

	
	model = Model()


	with sv.managed_session(config=config) as sess:
		for epoch in range(10):
			print("Epoch: ", epoch)
			if sv.should_stop():
				print('supervisor stoped!')
				break
			for step in range(num_tr_batch):
				start = step * batch_size
				end = start + batch_size
				global_step = epoch * num_tr_batch + step

				if sv.should_stop():
					print('supervisor stoped!')
					break

				### METHOD 1 ###
				### COMMENT/UNCOMMENT ###
				sess.run(model.train_op, {model.X:trX[start:end], model.Y:trY[start:end]})
				### END METHOD 1 ###

				### METHOD 2 ###
				### COMMENT/UNCOMMENT ###
				# sess.run(model.train_op)
				### END METHOD 2 ###

if __name__ == '__main__':
	main()

