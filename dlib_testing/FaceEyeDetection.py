
from copy import deepcopy
from collections import OrderedDict
import sys

import numpy as np
import matplotlib.pyplot as plt

import dlib
import cv2
import imutils


shape_predictor_68_face_landmarks = '/home/gsandh16/Downloads/shape_predictor_68_face_landmarks.dat'


class FaceEyeDetection(object):

	def __init__(self):

		# Face detector and landmark predictor.
		self.faceDetector = dlib.get_frontal_face_detector()
		self.landmarkPredictor = dlib.shape_predictor(shape_predictor_68_face_landmarks)
		self.faceDetectionPyramidFactor = 1
		self.FACIAL_LANDMARKS_IDXS = OrderedDict([
													("mouth", (48, 68)),
													("right_eyebrow", (17, 22)),
													("left_eyebrow", (22, 27)),
													("right_eye", (36, 42)),
													("left_eye", (42, 48)),
													("nose", (27, 36)),
													("jaw", (0, 17))
												])

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

		# Check to see if only one face was detected.
		if len(faceCoordsAll) != 1:
			print("Length of faceCoordsAll is ", len(faceCoordsAll))
			return False

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

	def showImagesOfFaceAndEyes(self, imageFile):

		faceRectangleExpansion = 0.20
		eyeRectangleHeightExpansion = 1.0
		eyeRectangleWidthExpansion = 0.3

		# Read in the image.
		image = self.readImage(imageFile)

		plt.imshow(image)
		plt.show()

		# Extract the face and eye coordinates.
		returnValue = self.extractSingleFaceAndEyeCoordsFromImage(image)
		if returnValue != False:
			faceCoords, leftEyeCoords, rightEyeCoords, binaryOverlap = returnValue
		else:
			return False

		# Expand the coordinates of the rectangles.
		faceCoords = self.expandCoords(faceCoords, faceRectangleExpansion, faceRectangleExpansion)
		leftEyeCoords = self.expandCoordsMakeRatioEven(leftEyeCoords, eyeRectangleWidthExpansion)
		rightEyeCoords = self.expandCoordsMakeRatioEven(rightEyeCoords, eyeRectangleWidthExpansion)

		# Get the subparts of the images corresponding to the rectangles.
		faceImage, leftEyeImage, rightEyeImage = self.getImagePartsFromCoords(image, faceCoords, leftEyeCoords, rightEyeCoords)

		# Resize.
		faceImage = cv2.resize(faceImage, (512, 512))
		leftEyeImage = cv2.resize(leftEyeImage, (256, 256))
		rightEyeImage = cv2.resize(rightEyeImage, (256, 256))

		plt.figure()
		plt.imshow(faceImage, cmap='jet')
		
		plt.figure()
		plt.imshow(leftEyeImage, cmap='gray')

		plt.figure()
		plt.imshow(rightEyeImage, cmap='gray')

		plt.show()

	def turnImageIntoSquare(self, image):
		### ASDF -> Untested for all combos of height/width (even/odd)
		height, width, depth = image.shape
		if height > width:
			difference = height-width
			if difference%2 == 0:
				image = image[difference//2:-difference//2, :]
			else:
				image = image[difference//2:-difference//2+1,:]
		elif height < width:
			difference = width-height
			if difference%2 == 0:
				image = image[:, difference//2:-difference//2]
				# image = image[height//2:-height//2, height//2:-height//2]
			else:				
				image = image[:, difference//2:-difference//2]

		return image

	def computeBinaryOverlap(self, image, faceCoords, leftEyeCoords=None, rightEyeCoords=None):
		""" Compute the overlap between the image and the face coordinates """

		# Obtain the shape.
		height, width, depth = image.shape

		# Allocate array of zeros.
		binaryOverlap = np.zeros((height, width, 1))

		# Obtain face coordinates.
		x = faceCoords[0]
		y = faceCoords[1]
		w = faceCoords[2]
		h = faceCoords[3]

		# Set the face coordinates to 1.
		binaryOverlap[y:y+h, x:x+w] = 1.0

		# if leftEyeCoords == None and rightEyeCoords == None:
		# 	return binaryOverlap
		# else:
		
		# 	# Obtain face coordinates.
		# 	x = leftEyeCoords[0]
		# 	y = leftEyeCoords[1]
		# 	w = leftEyeCoords[2]
		# 	h = leftEyeCoords[3]
		# 	print('left eye coords: ',leftEyeCoords)


		# 	# Set the face coordinates to 2.
		# 	binaryOverlap[y:y+h, x:x+w] = 1.0

		# 	# Obtain face coordinates.
		# 	x = rightEyeCoords[0]
		# 	y = rightEyeCoords[1]
		# 	w = rightEyeCoords[2]
		# 	h = rightEyeCoords[3]
		# 	print('right eye coords: ',rightEyeCoords)
		# 	print('x y w h',x, ' ',y,' ', w, ' ', h)
		# 	# Set the face coordinates to 2.
		# 	print('max value: ', np.max(binaryOverlap))
		# 	binaryOverlap[y:y+h, x:x+w] = 1.0
		# 	print('max value: ', np.max(binaryOverlap))

		return binaryOverlap

	def extractImagesOfFaceAndEyes(self, imageFile):

		faceRectangleExpansion = 0.20
		eyeRectangleHeightExpansion = 1.0
		eyeRectangleWidthExpansion = 0.3

		# Read in the image.
		image = self.readImage(imageFile)

		# Make the image square. Based on if the height or width is bigger, crop the image.
		image = self.turnImageIntoSquare(image)

		# Extract the face and eye coordinates.
		returnValue = self.extractSingleFaceAndEyeCoordsFromImage(image)
		if returnValue != False:
			faceCoords, leftEyeCoords, rightEyeCoords = returnValue
		else:
			return False

		# Compute the binary overlap.
		binaryOverlap = self.computeBinaryOverlap(image, faceCoords, leftEyeCoords, rightEyeCoords)

		# Expand the coordinates of the rectangles.
		faceCoords = self.expandCoords(faceCoords, faceRectangleExpansion, faceRectangleExpansion)
		leftEyeCoords = self.expandCoordsMakeRatioEven(leftEyeCoords, eyeRectangleWidthExpansion)
		rightEyeCoords = self.expandCoordsMakeRatioEven(rightEyeCoords, eyeRectangleWidthExpansion)

		# Get the subparts of the images corresponding to the rectangles.
		faceImage, leftEyeImage, rightEyeImage = self.getImagePartsFromCoords(image, faceCoords, leftEyeCoords, rightEyeCoords)

		# Resize.
		faceImage = cv2.resize(faceImage, (512, 512))
		leftEyeImage = cv2.resize(leftEyeImage, (256, 256))
		rightEyeImage = cv2.resize(rightEyeImage, (256, 256))
		binaryOverlap = cv2.resize(binaryOverlap, (16,16))

		return faceImage, leftEyeImage, rightEyeImage, binaryOverlap

	def loopThroughImagesUntilAssertionError(self, imageDir, numberImages):
		""" Tests to see how many images do not detect 1 face. """

		# Counters to see how many times extraction was or was not successful.
		numTrue = 0
		numFalse = 0

		# Loop though images until we reach the end.
		index = 0
		while index < numberImages:

			# Get the string of the picture index, i.e. '12456' for '12456.png'.
			numStr = '%05d'%index

			# Concatenate the image path.
			imageFile = imageDir + numStr + '.png'
			
			# Info.
			print(imageFile)

			# Extract images.
			returnValue = self.extractImagesOfFaceAndEyes(imageFile)

			# Increment counters to see if extraction was or was not sucessful.
			if returnValue == False:
				numFalse +=1
			else:
				numTrue +=1

			# Go to the next image.
			index +=1

			print("Current ratio -- true/false: ", numTrue/float(index),'/',numFalse/float(index))

	def saveFaceAndEyes(self, imageDir, outfileDir, numberImages):

		# Counters to see how many times extraction was or was not successful.
		numTrue = 0
		numFalse = 0

		# Loop though images until we reach the end.
		index = 0
		while index < numberImages:

			# Get the string of the picture index, i.e. '12456' for '12456.png'.
			numStr = '%05d'%index

			# Concatenate the image path.
			imageFile = imageDir + numStr + '.png'
			
			# Info.
			print(imageFile)

			# Extract images.
			returnValue = self.extractImagesOfFaceAndEyes(imageFile)

			# Increment counters to see if extraction was or was not sucessful.
			if returnValue == False:
				numFalse +=1
			else:
				numTrue +=1

				outfile = outfileDir + numStr + '.npz' 
				np.savez(outfile, faceImage=returnValue[0], leftEyeImage=returnValue[1], rightEyeImage=returnValue[2], binaryOverlap=returnValue[3])

			# Go to the next image.
			index +=1

			print("Current ratio -- true/false: ", numTrue/float(index),'/',numFalse/float(index))


	def getImagePartsFromCoords(self, image, faceCoords, leftEyeCoords, rightEyeCoords):
		""" For an image, get the different subimages of the face and eyes. """

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


	def expandCoordsMakeRatioEven(self, coords, xExpand):
		""" Obtain a larger bounding box as represented by the coordinates.
		    However, make the """

		# Get the coordinates.
		x = coords[0]
		y = coords[1]
		w = coords[2]
		h = coords[3]

		# Make the rectangle bigger to accomodate the area surrounding the eyes.
		xPlus = int(x+w+float(w)*xExpand)
		xMinus = int(x - float(w)*xExpand)
		wNew = xPlus-xMinus
		hNew = wNew

		if hNew <= h:
			print("The height must be greater than the current height.")
			sys.exit()

		difference = hNew-h

		if difference%2 == 0:
			yMinus = y - difference//2
			yPlus = y + h + difference//2
		else:
			yMinus = y - difference//2
			yPlus = y + h + difference//2 + 1

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





		 


