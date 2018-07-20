from FaceEyeDetection import FaceEyeDetection

def main():

	# imageFilePath = '/home/gsandh16/Documents/gazeTracking/data/unrealSyntheticImages/run2/00000.png'
	imageFilePath = '/home/gsandh16/Documents/gazeTracking/data/einstein.jpg'
	# imageFilePath = '/home/gsandh16/Documents/gazeTracking/data/mario.jpg'
	# imageFilePath = '/home/gsandh16/Documents/gazeTracking/data/lena.jpeg'
	# imageFilePath = '/home/gsandh16/Documents/gazeTracking/data/marilyn.jpg'

	faceEyeDetection = FaceEyeDetection()
	faceEyeDetection.detectAndDrawRectanglesOnFaces(imageFilePath)

if __name__ == '__main__':
	main()