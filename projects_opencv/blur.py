import cv2
import imutils
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def anonymize_face_simple(image, factor=5.0):
	# automatically determine the size of the blurring kernel based
	# on the spatial dimensions of the input image
	(h, w) = image.shape[:2]
	kW = int(w / factor)
	kH = int(h / factor)
	# ensure the width of the kernel is odd
	if kW % 2 == 0:
		kW -= 1
	# ensure the height of the kernel is odd
	if kH % 2 == 0:
		kH -= 1
	# apply a Gaussian blur to the input image using our computed
	# kernel size
	return cv2.GaussianBlur(image, (kW, kH), 0)


def anonymize_face_pixelate(image, blocks=15):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks + 1, dtype="int")
	ySteps = np.linspace(0, h, blocks + 1, dtype="int")
	# loop over the blocks in both the x and y direction
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# compute the starting and ending (x, y)-coordinates
			# for the current block
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]
			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image

			roi = image[startY:endY, startX:endX]
			#print(f"ROI {roi.shape}")
			#print(type(cv2.mean(roi)[:3]))
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)
	# return the pixelated blurred image
	return image


def forweb(im, face_cascade):
	#im = cv2.imread('pic.jpg')
	#im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

	
	
	gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
	faceRects = face_cascade.detectMultiScale(
		gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	for (fX, fY, fW, fH) in faceRects:
		# extract the face ROI
		faceROI = gray[fY:fY+ fH, fX:fX + fW]
		#cv2.rectangle(im, (fX, fY), (fX + fW, fY + fH),
		#		(0, 0, 255), 2)


	#crop = anonymize_face_simple(im[fY:fY+ fH, fX:fX + fW])
		crop = anonymize_face_pixelate(im[fY:fY+ fH, fX:fX + fW])
		im[fY:fY+ fH, fX:fX + fW] = crop
		cv2.imshow('im', im)
	#cv2.waitKey(0)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == '__main__':
	cam = cv2.VideoCapture(0)
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

	while True:
		ret, frame = cam.read()
		if not ret:
			raise Exception('Cannot take a shot')

		forweb(frame, face_cascade)
	
		
