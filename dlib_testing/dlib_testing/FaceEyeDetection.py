import sys
from copy import deepcopy

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

class FaceEyeDetection(object):

	def __init__(self):

		# Face detector and landmark predictor.
		self.faceDetector = dlib.get_frontal_face_detector()
		self.landmarkPredictor = dlib.shape_predictor(shape_predictor_68_face_landmarks)
		self.facePyramidLayers = 1
		self.horizontalExtend = 3.0
		self.verticalExtend = 1.0

	def readImage(self, imageFilePath):
		image = cv2.imread(imageFilePath)
		return image

	def manuallyCropImage(self, image, rowValues = [rowMin, rowMax], colValues = [colMin, colMax]):

		tmpImage = deepcopy(image)
		croppedImage = tmpImage[rowMin:rowMax,colMin,colMax]
		return croppedImage

	def resize(self, image, width):

		tmpImage = deepcopy(image)
		resizedImage = imutils.resize(image, width)
		return resizedImage

	def convertToGrayScale(self, image):
		grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		return grayImage

	def detectFaces(self, grayImage):
		faceCoordsAll = self.faceDetector(grayImage, self.facePyramidLayers)
		return faceCoordsAll

	def detectLandmarks(self, grayImage, faceCoords):

		assert len(faceCoords) == 1

		landmarks = self.predictor(grayImage, faceCoords)
		landmarks = helper_functions.rect_to_bb(faceCoords)

		return landmarks

	def getEyeCoords(self, landmarks, eye='left'):

		# Obtain the landmark indices based on if left or right eye.
		if eye == 'right':
			begin = 36
			end=42
		elif eye == 'left':
			begin=42
			end=48
		else:
			print("Incorrect eye type.")
			sys.exit()

		print('begin: ', begin, ' -- end: ', end)

		# Get the x and y values.
		xValues = []
		yValues = []
		for coordinate in landmarks[begin:end]:
			xValues.append(coordinate[0])
			yValues.append(coordinate[1])

		# Obtain the min/max values and the difference.
		maxX = np.max(xValues)
		minX = np.min(xValues)
		maxY = np.max(yValues)
		minY = np.min(yValues)
		dx = maxX-minX
		dy = maxY-minY

		# Scale the values based on how much we want to extend.
		maxX = int(maxX + dx/self.horizontalExtend)
		minX = int(minX - dx/self.horizontalExtend)
		maxY = int(maxY + dy/self.verticalExtend)
		minY = int(minY - dy/self.verticalExtend)
		dx = maxX - minX
		dy = maxY - minY

		return (minX, minY, dx, dy)









		 


