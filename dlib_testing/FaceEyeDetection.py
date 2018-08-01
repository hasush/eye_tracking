import sys
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import dlib
import cv2
import imutils



import helper_functions

shape_predictor_68_face_landmarks = '/home/gsandh16/Downloads/shape_predictor_68_face_landmarks.dat'


class FaceEyeDetection(object):

	def __init__(self):

		# Face detector and landmark predictor.
		self.faceDetector = dlib.get_frontal_face_detector()
		self.landmarkPredictor = dlib.shape_predictor(shape_predictor_68_face_landmarks)
		self.faceDetectionPyramidFactor = 1

	def readImage(self, imageFilePath):
		image = cv2.imread(imageFilePath)
		return image

	def manuallyCropImage(self, image, rowMinMax, colMinMax):

		rowMin = rowMinMax[0]
		rowMax = rowMinMax[1]
		colMin = colMinMax[0]
		colMax = colMinMax[1]

		tmp_image = deepcopy(image)
		croppedImage = tmp_image[rowMin:rowMax,colMin:colMax]
		return croppedImage

	def resizeImage(self, image, width, height):
		resizedImage = imutils.resize(image, width=width, height=height)
		return resizedImage

	def convertToGrayScale(self, image):
		grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		return grayImage

	def obtainFaceCoordsAll(self, grayImage):
		faceCoordsAll = self.faceDetector(grayImage, self.faceDetectionPyramidFactor)
		return faceCoordsAll

	def obtainLandMarksOnFace(self, grayImage, faceCoords):
		landmarks = self.landmarkPredictor(grayImage, faceCoords)
		return landmarks

	def obtainEyeCoords(self, landmarks, eye='right'):

		tmp_landmarks = deepcopy(landmarks)

		# Set the landmarks to the correct eye orientation.
		if eye == 'right':
			landmarks = tmp_landmarks[36:42]
		elif eye == 'left':
			landmarks = tmp_landmarks[42:48]
		else:
			print("Incorrect landmarks used for eyes.")
			sys.exit()

		# Initialize values for sides of the rectangles.
		leftX = 0
		rightX = 0
		leftY = 0
		rightY = 0

		# List for coordinates of the landmarks corresponding to the eyes.
		xValues = []
		yValues = []

		# Loop over the landmarks and extract the x and y coordinates.
		for coordinate in landmarks:
			xValues.append(coordinate[0])
			yValues.append(coordinate[1])

		# Obtain the min/max values and the difference.
		maxX = np.max(xValues)
		minX = np.min(xValues)
		maxY = np.max(yValues)
		minY = np.min(yValues)
		dx = maxX-minX
		dy = maxY-minY
		
		return (minX, minY, dx, dy)

	def drawRectanglesOnFacesAndEyes(self, imageFile):

		# Read in the image.
		image = self.readImage(imageFile)

		# Convert image to gray scale.
		grayImage = self.convertToGrayScale(image)

		# Get all of the detected faces within the image.
		faceCoordsAll = self.obtainFaceCoordsAll(grayImage)

		assert len(faceCoordsAll) > 0

		# Loop over all faces and show them.
		for i, faceCoords in enumerate(faceCoordsAll):

			# Obtain the landmarks on the face.
			landmarks = self.obtainLandMarksOnFace(grayImage, faceCoords)

			# Convert the face landmarks into (x,y) coordinates.
			landmarks = self.convertDlibLandmarksToCoords(landmarks)

			# Convert the face coords into rectangle coords.
			(x, y, w, h) = self.convertDlibRectangleToOpencvCoords(faceCoords)

			# Draw rectangle of face on image.
			cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)

			# Annotate the face.
			cv2.putText(image, "Face: {}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

			# Obtain the rectangle of the left eye and then draw it on the face.
			minX, minY, dx, dy = self.obtainEyeCoords(landmarks, 'left')
			cv2.rectangle(image, (minX, minY), (minX+dx, minY+dy), (255, 0, 0), 1)

			# Obtain the rectangle of the right eye and then draw it on the face.
			minX, minY, dx, dy = self.obtainEyeCoords(landmarks, 'right')
			cv2.rectangle(image, (minX, minY), (minX+dx, minY+dy), (255, 0, 0), 1)

			# Draw right eye landmarks.
			for (x, y) in landmarks[36:42]:
				cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

			# Draw left eye landmarks.
			for (x, y) in landmarks[42:48]:
				cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

		plt.imshow(image)
		plt.show()

	def extractSingleFaceAndEyeCoordsFromImage(self, image):
		"""
			Extract the coordinates of the face and the eyes from an image.

			@param image An image.
			@return The coordinates of the face, lefy eye, and right eye.
					Each item is a tuple of type (x, y, w, h), where x and y
					are matrix indices for opencv indexing typecalls and 
					w and h are the width and height of the rectangles.
		"""

		# Convert image to gray scale.
		grayImage = self.convertToGrayScale(image)

		# Get all of the detected faces within the image.
		faceCoordsAll = self.obtainFaceCoordsAll(grayImage)
		assert len(faceCoordsAll) == 1

		# Squeeze the array.
		print(faceCoordsAll)
		faceCoords = faceCoordsAll[0]

		# Obtain the landmarks on the face.
		landmarks = self.obtainLandMarksOnFace(grayImage, faceCoords)

		# Convert the face landmarks into (x,y) coordinates.
		landmarks = self.convertDlibLandmarksToCoords(landmarks)

		# Convert the face coords into rectangle coords.
		faceCoords = self.convertDlibRectangleToOpencvCoords(faceCoords)

		# Obtain the rectangle of the left eye.
		leftEyeCoords = self.obtainEyeCoords(landmarks, 'left')

		# Obtain the rectangle of the right eye and then draw it on the face.
		rightEyeCoords = self.obtainEyeCoords(landmarks, 'right')

		return faceCoords, leftEyeCoords, rightEyeCoords

	def getImagePartsFromCoords(self, image, faceCoords, leftEyeCoords, rightEyeCoords):

		# Make a copy of the image.
		tmp_image = deepcopy(image)
		image = tmp_image

		# Get the coordinates of the face.
		x = faceCoords[0]
		y = faceCoords[1]
		w = faceCoords[2]
		h = faceCoords[3]
		faceImage = image[y:y+h, x:x+w]

		# Get the coordinates of the face.
		x = leftEyeCoords[0]
		y = leftEyeCoords[1]
		w = leftEyeCoords[2]
		h = leftEyeCoords[3]
		leftEyeImage = image[y:y+h, x:x+w]

		# Get the coordinates of the face.
		x = rightEyeCoords[0]
		y = rightEyeCoords[1]
		w = rightEyeCoords[2]
		h = rightEyeCoords[3]
		rightEyeImage = image[y:y+h, x:x+w]

		return faceImage, leftEyeImage, rightEyeImage

	def expandCoords(self, coords, xExpand, yExpand):

		# Get the coordinates.
		x = coords[0]
		y = coords[1]
		w = coords[2]
		h = coords[3]

		# Make the rectangle bigger to accomodate the area surrounding the eyes.
		xPlus = int(x+w+float(w)*xExpand)
		yPlus = int(y+h+float(h)*yExpand)
		xMinus = int(x - float(w)*xExpand)
		yMinus = int(y - float(h)*yExpand)
		wNew = xPlus-xMinus
		hNew = yPlus-yMinus

		return (xMinus, yMinus, wNew, hNew)


	def convertDlibRectangleToOpencvCoords(self, rectangle):
		""" 
			Convert the member variables of Dlib's rectangle to
			coordinates, i.e. (x, y, w, h), which OpenCV uses.

			@param	rectangle Dlib rectangle
			@return (x,y,w,h)	x: x coordinate of rectangle.
								y: y coordinate of rectangle.
								w: width 
								h: height
								
		"""

		# Get the coordinates.
		x = rectangle.left()
		y = rectangle.top()
		w = rectangle.right() - x
		h = rectangle.bottom() - y

		return (x, y, w, h)

	def convertDlibLandmarksToCoords(self, landmarks, dtype="int"):
		"""
			Obtain the x,y coordinates of the landmarks and then set them to an
			array of tuples.

			@param landmarks The facial landmarks on the face region of an image.
			@return A list of (x,y) tuples corresponding to the coordinates of each landmark.
		"""

		# Allocate memory for the (x,y) coordinates.
		coordinates = np.zeros((68, 2), dtype=dtype)

		# Loop over landmarks and extract the x and y positions.
		for i in range(68):
			coordinates[i] = (landmarks.part(i).x, landmarks.part(i).y)

		return coordinates



		 


