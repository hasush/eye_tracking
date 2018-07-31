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

	def readImage(self, imageFilePath):
		image = cv2.imread(imageFilePath)
		return image

	def manuallyCropImage(self, image, (rowMin, rowMax), (colMin, colMax)):

		tmp_image = deepcopy(image)
		cropped_image = tmp_image[rowMin:rowMax,colMin,colMax]
		return cropped_image
		 


