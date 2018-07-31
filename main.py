import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from FaceEyeDetection import FaceEyeDetection

imageFilePath = '/home/gsandh16/Documents/gazeTracking/data/unrealSyntheticImages/run2/00000.png'
# imageFilePath = '/home/gsandh16/Documents/gazeTracking/data/einstein.jpg'
# imageFilePath = '/home/gsandh16/Documents/gazeTracking/data/mario.jpg'
# imageFilePath = '/home/gsandh16/Documents/gazeTracking/data/lena.jpeg'
# imageFilePath = '/home/gsandh16/Documents/gazeTracking/data/marilyn.jpg'

# Path to directory containing images whose faces and eyes will be extracted.
imageDirectory = '/home/gsandh16/Documents/gazeTracking/data/unrealSyntheticImages/run2/'

def determineAverageFaceEyeSize(imageDirectoryFilePath):
	""" Loop over all images within the directory and run facial and eye detection in
		order to determine the average size window for face and eye detection.

		@param imageDirectoryFilePath Path to the directory where images are contained.
	"""

	# Instantiate the face/eye detector.
	faceEyeDetection = FaceEyeDetection()

	# Obtain the contents of the directory.
	directoryList = os.listdir(imageDirectoryFilePath)

	# Containers to hold the dimensions of the images.
	eyeDimensions = []
	faceDimensions = []

	numberMatchingParams = 0
	numberNotMatchingParams = 0

	# Loop over the contents of the directory.
	for index, file in enumerate(directoryList[0:1000]):

		if index % 20 == 0:
			print("Iteration: ", index)

		# If the file is the json file which contains gaze information, continue.
		if '.json' in file:
			continue
		else:

			# Concatenate the file path.
			imageFilePath = imageDirectoryFilePath + file

			# Find the face and eyes in the image.
			foundFacesEyes = faceEyeDetection.determineNumFacesEyes(imageFilePath)

			if foundFacesEyes:
				numberMatchingParams +=1
			elif not foundFacesEyes:
				numberNotMatchingParams +=1


	print("Matching: ", numberMatchingParams)
	print("Not Matching: ", numberNotMatchingParams)




def main():

	

	

	faceEyeDetection = FaceEyeDetection()

	# # Find the face and eyes in the image.
	# foundFacesEyes = faceEyeDetection.detectionPipeLine(imageFilePath)

	# print(foundFacesEyes)

	# sys.exit()

	# faceEyeDetection.loadImageFromDisk(imageFilePath)
	# faceEyeDetection.convertBgrImageToGray()
	# faceEyeDetection.detectFaceAndEyes()

	# plt.figure()
	# plt.imshow(faceEyeDetection.facesRoiGray[0], cmap='gray')
	# plt.figure()
	# plt.imshow(faceEyeDetection.eyesRoiGray[0], cmap='gray')
	# plt.figure()
	# plt.imshow(faceEyeDetection.eyesRoiGray[1], cmap='gray')
	# plt.show()

	# faceEyeDetection.detectAndDrawRectanglesOnFaces(imageFilePath)


	

	determineAverageFaceEyeSize(imageDirectory)

	








if __name__ == '__main__':
	main()