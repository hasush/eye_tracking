import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import dlib
import cv2
import imutils

dataDirPath = os.path.normpath('/home/gsandh16/Documents/gazeTracking/data/unrealSyntheticImages/run2_extracted/')

from FaceEyeDetection import FaceEyeDetection

class DataUtils(object):

	def __init__(self, imageInputDir, labelInputDir, dataFileDir):
		self.imageInputDir = imageInputDir
		self.labelInputDir = labelInputDir
		self.dataFileDir = dataFileDir
		self.dataX = []
		self.dataY = []

	def loadFaceEyesLabelsSmallFiles(self, save=True):
		""" Load images of face and eyes and print out batches of images
		 	as individual numpy files. Also print out corresponding labels 
		 	as batches. 

			@param save Whether to save the data or not.
		"""

		# Directory path to where the images are located.
		imageInputDir = os.path.normpath(self.imageInputDir)

		# List contents of the image directroy and then sort the list.
		dirContents = os.listdir(imageInputDir)
		dirContents.sort()

		# List conents of the json label directory.
		labelInputDir = os.path.normpath(self.labelInputDir)

		# The data will be stored.
		self.dataX = []
		self.dataY = []

		# Initialize a counter for image index.
		counter = 1

		# Initialize a counter for batch index.
		batch = 0

		# The batch size.
		batchSize=128

		# The number of images we will extract.
		numberImages = 1024

		# Loop over the files in the image directory.
		for file in dirContents:

			# If we have created the number of images requested, then break the loop.
			if counter > numberImages:
				break

			# If the file in the directory has the proper extension, read it.
			if '.npz' in file:
				print(file)

				# Concatenate the directory path with the name of the file.
				imageDataPath = os.path.join(imageInputDir, file)

				# Load the data.
				data = np.load(imageDataPath)

				# Get the images of the various sub regions of the image.
				faceImage = data['faceImage']
				leftEyeImage = data['rightEyeImage']
				rightEyeImage = data['rightEyeImage']
				binaryOverlap = data['binaryOverlap']

				# Normalize the images.
				newFaceImage = self.normalizeImage(faceImage)
				newLeftEyeImage = self.normalizeImage(leftEyeImage)
				newRightEyeImage = self.normalizeImage(rightEyeImage)

				# Append the images as a tuple of the set of sub regions.
				self.dataX.append((newFaceImage, newLeftEyeImage, newRightEyeImage, binaryOverlap))

				# Extract the sample number of this file.
				file = file.partition('.npz')

				# Convert the file name into the corresponding json file with label.
				file = file[0] + '.json'

				# Concatenate the json file directory paht with the json file.
				jsonFilePath = os.path.join(labelInputDir, file)

				print('json path: ',jsonFilePath)
				print('image path: ', imageDataPath)

				# Open and load the json file and then append it to the label data.
				with open(jsonFilePath) as jsonFile:
					data = json.load(jsonFile)
					self.dataY.append(data)

			# We have gotten all the images needed for a batch.
			if save and counter%batchSize == 0:

				# Get the path to where we would like to dump the data.
				dataFileDir = os.path.normpath(self.dataFileDir)
				print(dataFileDir)

				# Concatenate directory path with the label of the batch and X data.
				filePath = os.path.join(dataFileDir, 'dataX_batch_' + str(batch) + '.npy')

				print(filePath)

				# Save the imaging X data.
				np.save(filePath, self.dataX)

				# Concatenate directory path with the label of the batch and Y data.
				filePath = os.path.join(dataFileDir,'dataY_batch_' + str(batch) + '.npy')

				# Save the the head/pose/etc. Y data.
				np.save(filePath, self.dataY)

				# Increment the batch size.
				batch+=1

				# Reset the data storages.
				self.dataX = []
				self.dataY = []

			# Increment the counter.
			counter+=1

	def normalizeImage(self, image):
		""" Normalize the image pixels. 
			@param image The image to be normalized.

			@return A normalized image.
		"""

		# Choose normalization metho. TO DO: Put into config file.
		# return self.meanSubtractChannelNormalizeImage(image)
		# return self.rangeZeroToOneNormalizeImage(image)
		return self.divideBy255NormalizeImage(image)

	def meanSubtractChannelNormalizeImage(self, image):
		"""	Subtract the mean from each channel of the image.
			@param image the Image to be normalized.

			@return newImage The normalized image.
		"""

		# Obtain shape of the image.
		shape = image.shape

		# Allocate memory for the new image that will be created.
		newImage = np.zeros((shape[0], shape[1], shape[2]))
		
		# Obtain the channels of the image.
		zeroChannel = image[:,:,0]
		oneChannel = image[:,:,1]
		twoChannel = image[:,:,2]

		# Subtract the mean of each channel of the image.
		newImage[:,:,0] = zeroChannel - np.mean(zeroChannel)
		newImage[:,:,1] = oneChannel - np.mean(oneChannel)
		newImage[:,:,2] = twoChannel - np.mean(twoChannel)

		return newImage

	def rangeZeroToOneNormalizeImage(self, image):
		"""	Make the pixel values range from 0 to 1.
			@param image the Image to be normalized.

			@return newImage The normalized image.
		"""

		# Obtain shape of the image.
		shape = image.shape

		# Allocate memory for the new image that will be created.
		newImage = np.zeros((shape[0], shape[1], shape[2]))
		
		# Obtain the channels of the image.
		zeroChannel = image[:,:,0]
		oneChannel = image[:,:,1]
		twoChannel = image[:,:,2]

		# Subtract the mean of each channel of the image.
		newImage[:,:,0] = (zeroChannel - np.min(zeroChannel))/(np.max(zeroChannel)-np.min(zeroChannel))
		newImage[:,:,1] = (oneChannel - np.min(oneChannel))/(np.max(oneChannel)-np.min(oneChannel))
		newImage[:,:,2] = (twoChannel - np.min(twoChannel))/(np.max(twoChannel)-np.min(twoChannel))

		return newImage

	def divideBy255NormalizeImage(self, image):
		"""	Normalize image by dividing each pixel by 255.
			@param image the Image to be normalized.

			@return newImage The normalized image.
		"""

		# Obtain shape of the image.
		shape = image.shape

		# Allocate memory for the new image that will be created.
		newImage = np.zeros((shape[0], shape[1], shape[2]))

		newImage = image/255.
		return newImage

	def loadLabels(self):

		labelInputDir = os.path.normpath(self.labelInputDir)
		dirContents = os.listdir(labelInputDir)

		self.dataY = []
		for file in dirContents:
			print(file)
			if '.json' in file:

				filePath = os.path.join(labelInputDir, file)

				with open(filePath) as jsonFile:
					data = json.load(jsonFile)
					self.dataY.append(data)
				# faceImage = data['faceImage']
				# leftEyeImage = data['rightEyeImage']
				# rightEyeImage = data['rightEyeImage']
				# binaryOverlap = data['binaryOverlap']

				
				# newFaceImage = self.meanSubtractChannelNormalizeImage(faceImage)
				# newLeftEyeImage = self.meanSubtractChannelNormalizeImage(leftEyeImage)
				# newRightEyeImage = self.meanSubtractChannelNormalizeImage(rightEyeImage)

				
				sys.exit()



