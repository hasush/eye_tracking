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
image_file = '/home/gsandh16/Documents/gazeTracking/data/unrealSyntheticImages/run2/00000.png'



def main():

	# Face detector and landmark predictor.
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor_68_face_landmarks)

	# Read in the image.
	image = cv2.imread(image_file)

	# Manually crop the image.
	image = image[200:500, 400:800]
	plt.figure()
	plt.imshow(image)
	
	# Resize into a size of 500.
	image = imutils.resize(image, width=512)
	# print(image.shape)
	# plt.figure()
	# plt.imshow(image)
	# plt.show()

	# Convert image to gray scale.
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

		landmarks = predictor(gray, faceCoords)
		landmarks = helper_functions.shape_to_np(landmarks)

		(x,y, w, h) = helper_functions.rect_to_bb(faceCoords)
		cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


		# Right eye.
		leftX = 0
		rightX = 0
		leftY = 0
		rightY = 0
		# for coordinate in landmarks:

		print(landmarks[36:42][0])
		print('\nasdf\n')
		xValues = []
		yValues = []
		for coordinate in landmarks[36:42]:
			print(coordinate)
			xValues.append(coordinate[0])
			yValues.append(coordinate[1])

		maxX = np.max(xValues)
		minX = np.min(xValues)
		maxY = np.max(yValues)
		minY = np.min(yValues)
		dx = maxX-minX
		dy = maxY-minY

		print('minY: ', minY, ' -- minX: ', minX, ' -- dx: ',dx, ' -- dy: ', dy)

		maxX = int(maxX + dx/3.0)
		minX = int(minX - dx/3.0)
		maxY = int(maxY + dy/1.0)
		minY = int(minY - dy/1.0)

		dx = maxX - minX
		dy = maxY - minY
		print('minY: ', minY, ' -- minX: ', minX, ' -- dx: ',dx, ' -- dy: ', dy)
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

