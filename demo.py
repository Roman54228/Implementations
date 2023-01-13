import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import cv2
import numpy as np
import os
import argparse
from torchvision.utils import save_image

import os
import torch
import torch.nn as nn
from common import get_points_from_file_name
from face_attribute_onnx import FaceAttribute
from time import time
import argparse
from pathlib import Path
from collections import deque
import timm
import onnxruntime
from detector_dgx import get_detector
import glob


def read_img(flp):
	img = cv2.imread(flp)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	return img 


def crop_x2_replicate_300(img, coords):
	x, y, x1, y1 = coords 
	dh, dw, ch = img.reshape

	w = x1 - x
	h = y1 - y
	wb = w // 2
	hb = h // 2

	top_board = min(0, y - hb) * -1
	left_board = min(0, x - wb) * -1
	bottom_board = min(0, dh - (y1 + hb)) * -1
	right_board = min(0, dw - (x1 + wb)) * -1 

	crop = img[max(0-yhb):min(dh,y1+hb), max(0,x-wb):min(dw,x1+wb)]
	border_crop = cv2.copyMakeBorder(crop, top=top_board, bottom=bottom_board,
									left=left_board, right=right_board, borderType=cv2.BORDER_REPLICATE)
	return cv2.resize(border_crop, (300,300))


def process_from_video(detector, clahe, video, display, to_save=False):
	if video == 'web':
		video = 0
	camera = cv2.VideoCapture(video)

	#result_videos_dir = 'result_videos/'
	#ret, frame = camera.read()
	#height, width, layers = frame.shape
	#size = (width,height)
	size = (500,650)

	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	out = cv2.VideoWriter('video.mp4',fourcc, 20.0, size)

	if (camera.isOpened()== False): 
  		print("Error opening video stream or file")

	while(camera.isOpened()):
		ret, bgr = camera.read()
		if ret: 
			if detector is None:
				pass
			else:
				try:
					probs, faces_coord, face_points, angles = detector(bgr)
					if len(faces_coord) == 1:
						x, y, x1, y1 = faces_coord[0]
						size = x1 - x
						points = face_points[0]
						bgr = cv2.resize(bgr[y:y1, x:x1, :], (300,300))
				except:
					pass

			lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
			l, a, b = cv2.split(lab)
			clahe_img = clahe.apply(l)	
			updated_lab_img2 = cv2.merge((clahe_img,a,b))
			clahe_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
			cv2.putText(bgr, 'Original', (20,20), cv2.FONT_HERSHEY_SIMPLEX,
			 		1, (0,0,255), 2, cv2.LINE_AA)
			cv2.putText(clahe_img, 'CLAHE', (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
			 		1, (0,0,255), 2, cv2.LINE_AA)
			h_img = cv2.hconcat([bgr, clahe_img])

			if True:
				out.write(h_img)
			if display:
				cv2.imshow('resutls', h_img)
				cv2.waitKey(1)
		else:
			break
		

def process_from_1photo(image_path, detector, clahe):

	source_image = cv2.imread(image_path)
	lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)		
	clahe_img = clahe.apply(l)	
	updated_lab_img2 = cv2.merge((clahe_img,a,b))
	clahe_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)

	cv2.putText(source_image, 'Original', (20,20), cv2.FONT_HERSHEY_SIMPLEX,
				1, (0,0,255), 2, cv2.LINE_AA)
	cv2.putText(clahe_img, 'CLAHE', (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
				1, (0,0,255), 2, cv2.LINE_AA)
	return source_image, clahe_img


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--video', required=True, default='web')
	parser.add_argument('--detector', action="store_true", default=False)
	parser.add_argument('--display', action="store_true", default=False)
	parser.add_argument('--save_video', action="store_true", default=False)
	parser.add_argument('--image_path', default=None)
	args = parser.parse_args()

	image_path = args.image_path
	if args.detector:
		det = get_detector(Path('cfg_mnet68_angles_c_320_01_c.onnx'), 320, 0.5)
	else:
		det = None
	clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(5,5))
	
	source_dir = 'source_dir'
	out_dir = 'out_dir'

	if os.path.isdir(args.video) and args.video != 'web':
		for im_p in glob.glob(args.video+'/*'):
			
			source_img, clahe_img = process_from_1photo(im_p, det, clahe)
			h_img = cv2.hconcat([source_img, clahe_img])
			cv2.imwrite(os.path.join(out_dir, 'clahe_'+im_p.split('/')[-1]), h_img)

	else:
		process_from_video(det, clahe, args.video, args.display, args.save_video)

	# elif image_path:
	# 	if not os.path.exists(image_path):
	# 		raise ValueError('No such file')

	# 	source_img, clahe_img = process_from_1photo(image_path, det, clahe)
		
	# 	cv2.imshow('orig', source_img)
	# 	cv2.imshow('clahe', clahe_img)
	# 	cv2.waitKey()





		
