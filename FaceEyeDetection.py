import numpy as np
import cv2
import matplotlib.pyplot as plt

scaleFactorFaces = 1.01
minNeighborsFaces = 4 #4

scaleFactorEyes = 1.2 #1.2/5=301 or 1.1=291, 1.3=251, 1.25=245, 1.09=277, 1.21=293, 1.19=292
minNeighborsEyes = 5 #with 1.2: 4=298, 3=281, 2=256, 6 = 300, 7 =293

faceRectangleColor = (255,0,0)
faceRectangleThickness = 2
eyeRectangleColor = (0,255,0)
eyeRectangleThickness = 2
opencvDataPath = '/home/gsandh16/pythonEnvs/eye_tracking/lib/python3.6/site-packages/cv2/data/'
cropFaces = True

class FaceEyeDetection(object):
	""" Class for face and eye detection """
	
	def __init__(self):
		""" Instantiate the class """

		# Load the Haar-cascade based face and eye classifiers.
		self.face_cascade = cv2.CascadeClassifier(opencvDataPath + 'haarcascade_frontalface_default.xml')
		self.eye_cascade = cv2.CascadeClassifier(opencvDataPath + 'haarcascade_eye.xml')
		self.facesCoords = []
		self.eyesCoords = []
		self.facesRoiGray = []
		self.facesRoiColor = []
		self.eyesRoiGray = []
		self.eyesRoiColor = []
		self.image = None
		self.imageGray = None

	def loadImageFromDisk(self, filePath):
		""" Load an iamge from disk and set to a member variable.
			@param filePath File path of the image that will be read.
		"""

		print(filePath)

		# Read the image.
		self.image = cv2.imread(filePath)

		# plt.imshow(self.image)
		# plt.show()

		# print(type(self.image))
		# print(self.image.shape)

		if cropsFaces:
			self.image = self.image[150:, 400:800, :]

	def convertBgrImageToGray(self):
		""" Convert an BGR image to a gray scale image """

		# Convert the image.
		self.imageGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		
	def detectFaces(self):
		""" Function to detect face given that it exists on disk. """

		# Detect faces within the image.
		self.facesCoords = self.face_cascade.detectMultiScale(self.imageGray, scaleFactorFaces, minNeighborsFaces)

		# Check to make sure a face was detected.
		if len(self.facesCoords) < 1:
			return False
		else:
			return True

	def detectEyesInFace(self):
		""" Detect eyes contained in the faces """

		# Check to make sure only one face.
		# assert len(self.facesCoords) == 1
		if len(self.facesCoords) < 1:
			return

		# Reset the face ROIs and eyes.
		self.eyesCoords = []
		self.facesRoiGray = []
		self.facesRoiColor = []

		# Obtain coords of the face.
		(x,y,w,h) = self.facesCoords[0]

		# Get ROIs of the face.
		self.facesRoiGray.append(self.imageGray[y:y+h, x:x+w])
		self.facesRoiColor.append(self.image[y:y+h, x:x+w])

		# Find the eyes of the current ROI.
		self.eyesCoords.append(self.eye_cascade.detectMultiScale(self.facesRoiGray[0], scaleFactorEyes, minNeighborsEyes))

		# Make sure only one set of eyes and that the set contains two eyes.
		if len(self.eyesCoords) != 1 or len(self.eyesCoords[0]) != 2:
			self.facesRoiGray.pop()
			self.facesRoiColor.pop()
			return

		assert len(self.eyesCoords) == 1
		assert len(self.eyesCoords[0]) == 2

		# Loop over coordinates of the eyes.
		for (ex, ey, ew, eh) in self.eyesCoords[0]:

			# Get the ROIs of the eyes.
			self.eyesRoiGray.append(self.imageGray[y+ey:y+ey+eh,x+ex:x+ex+ew])
			self.eyesRoiColor.append(self.image[y+ey:y+ey+eh,x+ex:x+ex+ew])

	def drawRectanglesOnEyesAndFace(self):
		""" Detect eyes contained in the faces """

		# Reset the face ROIs and eyes.
		self.eyesCoords = []
		self.facesRoiGray = []
		self.facesRoiColor = []

		# Loop over the coordinates of the faces.
		for (x,y,w,h) in self.facesCoords:

			# Draw a rectangle on the face.
			cv2.rectangle(self.image, (x, y), (x+w, y+h), faceRectangleColor, faceRectangleThickness)

			# Get ROIs of the face.
			self.facesRoiGray.append(self.imageGray[y:y+h, x:x+w])
			self.facesRoiColor.append(self.image[y:y+h, x:x+w])

			# Find the eyes of the current ROI.
			self.eyesCoords.append(self.eye_cascade.detectMultiScale(self.facesRoiGray[-1], scaleFactorEyes, minNeighborsEyes))

			assert len(self.eyesCoords) == 1

			# Loop over coordinates of the eyes.
			for (ex, ey, ew, eh) in self.eyesCoords[-1]:

				# Draw rectangles on the eyes.
				cv2.rectangle(self.facesRoiColor[-1], (ex, ey), (ex+ew, ey+eh), eyeRectangleColor, eyeRectangleThickness)

	def detectAndDrawRectanglesOnFaces(self, filePath):
		""" Draw rectangles on faces and eyes within an image contained
			@param filePath File path of the iamge that will be read.
		"""

		self.loadImageFromDisk(filePath)
		self.convertBgrImageToGray()
		self.detectFaces()
		self.drawRectanglesOnEyesAndFace()

		plt.imshow(self.image)
		plt.show()

	def detectFaceAndEyes(self):
		self.detectFaces()
		self.detectEyesInFace()

	def determineNumFacesEyes(self, filePath):

		# Read the image.
		self.image = cv2.imread(filePath)

		# Convert the image.
		self.imageGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

		# Detect faces within the image.
		self.facesCoords = self.face_cascade.detectMultiScale(self.imageGray, scaleFactorFaces, minNeighborsFaces)

		# Check to make sure one face was detected.
		if len(self.facesCoords) != 1:
			return False

		# Obtain coords of the face.
		(x,y,w,h) = self.facesCoords[0]

		# Get ROIs of the face.
		self.facesRoiGray = self.imageGray[y:y+h, x:x+w]
		self.facesRoiColor = self.image[y:y+h, x:x+w]

		# Find the eyes of the current ROI.
		self.eyesCoords = self.eye_cascade.detectMultiScale(self.facesRoiGray, scaleFactorEyes, minNeighborsEyes)

		# Make sure only one set of eyes and that the set contains two eyes.
		if len(self.eyesCoords) != 2:
			return False
		else:
			return True

		# Coordinates of the first eye.
		(ex1, ey1, ew1, eh1) = self.eyesCoords[0]

		# Extrat images of the first eye.
		self.eyesRoiGray1 = self.imageGray[y+ey1:y+ey1+eh1,x+ex1:x+ex1+ew1]
		self.eyesRoiColor1 = self.image[y+ey1:y+ey1+eh1,x+ex1:x+ex1+ew1]

		# Coordinates of the second eye.
		(ex2, ey2, ew2, eh2) = self.eyesCoords[1]

		# Extract images of the second eye.
		self.eyesRoiGray2 = self.imageGray[y+ey2:y+ey2+eh2,x+ex2:x+ex2+ew2]
		self.eyesRoiColor2 = self.image[y+ey2:y+ey2+eh2,x+ex2:x+ex2+ew2]

		return True