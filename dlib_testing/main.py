import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

import dlib
import cv2
import imutils


from FaceEyeDetection import FaceEyeDetection


# image_file = '/home/gsandh16/Documents/gazeTracking/data/einstein.jpg'
imageFile = '/home/gsandh16/Documents/gazeTracking/data/unrealSyntheticImages/run2/00063.png'
# imageFile = '/home/gsandh16/Documents/gazeTracking/data/randomFamily.jpeg'

faceRectangleExpansion = 0.20
eyeRectangleHeightExpansion = 1.0
eyeRectangleWidthExpansion = 0.33

def main():

	# Instantiate facial eye detection.
	fed = FaceEyeDetection()

	# # Draw rectangles on the faces and eyes within an image.
	# fed.drawRectanglesOnFacesAndEyes(imageFile)

	# # Extract face and eye regions and then display.
	# fed.showImagesOfFaceAndEyes(imageFile)

	# fed.loopThroughImagesUntilAssertionError('/home/gsandh16/Documents/gazeTracking/data/unrealSyntheticImages/run2/', 39006)

	# returnValue = fed.extractImagesOfFaceAndEyes(imageFile)

	# print(type(returnValue[0]))
	# print(returnValue[0].shape)

	# plt.imshow(returnValue[1])
	# plt.show()

	# outfile = '/home/gsandh16/Documents/gazeTracking/data/unrealSyntheticImages/test.npz' 
	# np.savez(outfile, faceImage=returnValue[0], leftEyeImage=returnValue[1], rightEyeImage=returnValue[2])

	# asdf = np.load(outfile)

	# plt.imshow(asdf['faceImage'])
	# plt.show()

	imageDir = '/home/gsandh16/Documents/gazeTracking/data/unrealSyntheticImages/run2/'
	outfileDir = '/home/gsandh16/Documents/gazeTracking/data/unrealSyntheticImages/run2_extracted/'
	fed.saveFaceAndEyes(imageDir, outfileDir, 10)
if __name__ == '__main__':
	main()

