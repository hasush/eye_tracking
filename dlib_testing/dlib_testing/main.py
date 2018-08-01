import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

import dlib
import cv2
import imutils

# from imutils import face_utils

import helper_functions

shape_predictor_68_face_landmarks = '/home/gsandh16/Downloads/shape_predictor_68_face_landmarks.dat'
# image_file = '/home/gsandh16/Documents/gazeTracking/data/einstein.jpg'
imageFile = '/home/gsandh16/Documents/gazeTracking/data/unrealSyntheticImages/run2/00000.png'

from FaceEyeDetection import FaceEyeDetection


def main():

	# Instantiate the facial and eye detection.
	fed = FaceEyeDetection()


	# Read in the image.	
	image = fed.readImage(imageFile)

	# Manually crop the image.
	image = fed.manuallyCropImage(image, rowValues = [200,500], colValues = [400,800])
	plt.figure()
	plt.imshow(image)
	plt.show()
	
	# Resize into a size of 512.
	image = fed.resize(image, width=512)

	# Convert image to gray scale.
	gray = fed.convertToGrayScale(image)

	# Get all of the detected faces within the image.
	faceCoordsAll = detector(gray, 1)

	print(faceCoordsAll)
	

	if len(faceCoordsAll) < 2:
		pass

	# Loop over all faces and show them.
	for i, faceCoords in enumerate(faceCoordsAll):
		face = image[faceCoords.top():faceCoords.bottom(),faceCoords.left():faceCoords.right()]

		# plt.figure()
		# plt.imshow(face)
		# plt.show()

		landmarks = fed.detectLandmarks(gray, faceCoords)

		(x,y, w, h) = helper_functions.rect_to_bb(faceCoords)
		cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


		minX,minY,dx,dy = fed.getEyeCoords(landmarks, eye='left')
		cv2.rectangle(image, (minX, minY), (minX+dx,minY+dy), (255,0,0), 1)

	

		for (x, y) in landmarks[36:42]:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

		# Left eye.
		for (x, y) in landmarks[42:48]:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


	plt.imshow(image)
	plt.show()



	# face = image[rects[0]:rects[1]]

if __name__ == '__main__':
	main()

