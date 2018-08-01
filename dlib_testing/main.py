import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

import dlib
import cv2
import imutils


from FaceEyeDetection import FaceEyeDetection


# image_file = '/home/gsandh16/Documents/gazeTracking/data/einstein.jpg'
imageFile = '/home/gsandh16/Documents/gazeTracking/data/unrealSyntheticImages/run2/00000.png'
# imageFile = '/home/gsandh16/Documents/gazeTracking/data/randomFamily.jpeg'

faceRectangleExpansion = 0.4
eyeRectangleHeightExpansion = 1.0
eyeRectangleWidthExpansion = 0.33

def main():

	# Instantiate facial eye detection.
	fed = FaceEyeDetection()

	# # Draw rectangles on the faces and eyes within an image.
	# fed.drawRectanglesOnFacesAndEyes(imageFile)

	# Read in the image.
	image = fed.readImage(imageFile)


	faceCoords, leftEyeCoords, rightEyeCoords = fed.extractSingleFaceAndEyeCoordsFromImage(image)

	faceCoords = fed.expandCoords(faceCoords, faceRectangleExpansion, faceRectangleExpansion)
	leftEyeCoords = fed.expandCoords(leftEyeCoords, eyeRectangleWidthExpansion, eyeRectangleHeightExpansion)
	rightEyeCoords = fed.expandCoords(rightEyeCoords, eyeRectangleWidthExpansion, eyeRectangleHeightExpansion)

	faceImage, leftEyeImage, rightEyeImage = fed.getImagePartsFromCoords(image, faceCoords, leftEyeCoords, rightEyeCoords)

	faceImage = fed.resizeImage(faceImage, 512, 512)
	leftEyeImage = cv2.resize(leftEyeImage, (256, 171))
	rightEyeImage = cv2.resize(rightEyeImage, (256, 171))

	plt.figure()
	plt.imshow(faceImage)
	
	plt.figure()
	plt.imshow(leftEyeImage)

	plt.figure()
	plt.imshow(rightEyeImage)

	plt.show()

if __name__ == '__main__':
	main()

