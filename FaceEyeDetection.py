import numpy as np
import cv2
import matplotlib.pyplot as plt

scaleFactor = 1.01
minNeighbors = 5
faceRectangleColor = (255,0,0)
faceRectangleThickness = 2
eyeRectangleColor = (0,255,0)
eyeRectangleThickness = 2
opencvDataPath = '/home/gsandh16/pythonEnvs/eye_tracking/lib/python3.6/site-packages/cv2/data/'

class FaceEyeDetection(object):
	""" Class for face and eye detection """
	
	def __init__(self):
		""" Instantiate the class """

		# Load the Haar-cascade based face and eye classifiers.
		self.face_cascade = cv2.CascadeClassifier(opencvDataPath + 'haarcascade_frontalface_default.xml')
		self.eye_cascade = cv2.CascadeClassifier(opencvDataPath + 'haarcascade_eye.xml')
		self.faces = []
		self.eyes = []
		self.faceRoiGray = []
		self.faceRoiColor = []
		self.image = None
		self.imageGray = None
		self.faceImages = None
		self.eyeImages = None

	def loadImageFromDisk(self, filePath):
		""" Load an iamge from disk and set to a member variable.
			@param filePath File path of the image that will be read.
		"""

		# Read the image.
		self.image = cv2.imread(filePath)

		print(type(self.image))
		print(self.image.shape)

		# self.image = self.image[150:, 400:800, :]

	def convertBgrImageToGray(self):
		""" Convert an BGR image to a gray scale image """

		# Convert the image.
		self.imageGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

		plt.imshow(self.imageGray)
		plt.show()
		
	def detectFaces(self):
		""" Function to detect face given that it exists on disk. """

		# Detect faces within the image.
		self.faces = self.face_cascade.detectMultiScale(self.imageGray, scaleFactor, minNeighbors)

		assert len(self.faces) > 0

	def detectEyesInFaces(self):
		""" Detect eyes contained in the faces """

		# Reset the face ROIs and eyes.
		self.eyes = []
		self.faceRoiGray = []
		self.faceRoiColor = []

		# Loop over the coordinates of the faces.
		for (x,y,w,h) in self.faces:


			plt.imshow(self.image[y:y+h, x:x+w])
			plt.show()
			
			# Draw a rectangle on the face.
			cv2.rectangle(self.image, (x, y), (x+w, y+h), faceRectangleColor, faceRectangleThickness)

			


			# Get ROIs of the images.
			self.faceRoiGray.append(self.imageGray[y:y+h, x:x+w])
			self.faceRoiColor.append(self.image[y:y+h, x:x+w])

			# Find the eyes of the current ROI.
			self.eyes.append(self.eye_cascade.detectMultiScale(self.faceRoiGray[-1]))

			assert len(self.eyes) > 0

			# Draw rectangles for the eyes.
			for (ex, ey, ew, eh) in self.eyes[-1]:

				# Draw rectangles on the eyes.
				cv2.rectangle(self.faceRoiColor[-1], (ex, ey), (ex+ew, ey+eh), eyeRectangleColor, eyeRectangleThickness)

	def detectAndDrawRectanglesOnFaces(self, filePath):
		""" Draw rectangles on faces and eyes within an image contained
			@param filePath File path of the iamge that will be read.
		"""

		self.loadImageFromDisk(filePath)
		self.convertBgrImageToGray()
		self.detectFaces()
		self.detectEyesInFaces()

		plt.imshow(self.image)
		plt.show()
		# cv2.imshow('image', self.image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()













		
