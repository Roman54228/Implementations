from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as rt

import mytypes as t


def prepare_session(model_path: Path) -> rt.InferenceSession:
    return rt.InferenceSession(str(model_path))


def get_detector(model_path: Path, detector_shape: int,  threshold: int, lands: bool=False) -> t.Detector:
    rt_sess = prepare_session(model_path)
    input_name = rt_sess.get_inputs()[0].name
    rgb_mean = (117., 123., 104.)

    def prepare_img(img: t.RGBImg) -> Tuple[t.RGBImg, int]:
        new_img = np.zeros((detector_shape, detector_shape, 3), dtype=img.dtype)
        height, width = img.shape[:2]
        scale = detector_shape / max(height, width)

        img = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        new_height, new_width = img.shape[:2]
        new_img[:new_height, :new_width] = img
        new_img = new_img.astype(np.float32)
        new_img -= rgb_mean
        new_img = new_img.transpose((2, 0, 1))[np.newaxis, ...]
        return new_img, max(width, height),scale

    def detect(img: t.RGBImg) -> Tuple[t.Scores, t.Bboxes, t.Landmarks, t.Additional]:
        img, scale,true_scale = prepare_img(img)
        scores, bboxes, landmarks, angles = rt_sess.run(None, {input_name: img})

        scor = []
        bbox = []
        face_points = []
        angle = []

        if len(scores) > 0:
            # print(bboxes)
            # print(bboxes*true_scale,true_scale)

            bboxes *= scale
            # print(bboxes, scale)
            landmarks *= scale
            landmarks = np.reshape(landmarks, (bboxes.shape[0], -1, 2))

        for i in range(len(scores)):
            if scores[i]>threshold:
                scor.append(scores[i])
                angle.append(angles[i])

                left_eye = np.mean(landmarks[i][36:42],axis=0)
                right_eye = np.mean(landmarks[i][42:48],axis=0)
                nose = np.mean(landmarks[i][30:31],axis=0)
                left_mouth = np.mean(landmarks[i][48:49],axis=0)
                right_mouth = np.mean(landmarks[i][54:55],axis=0)  
                face_points.append(np.stack((left_eye,right_eye,nose,left_mouth,right_mouth)).astype(int).T.flatten())

                x,y,x1,y1 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
                center_x = (x1+x)//2
                center_y = (y1+y)//2
                half_border = max(x1-x,y1-y)//2 
                x = center_x - half_border
                x1 = center_x + half_border
                y = center_y - half_border
                y1 = center_y + half_border 
                bbox.append(np.array([x,y,x1,y1],dtype=int))

        return np.array(scor), np.array(bbox), np.array(face_points), np.array(angle)
    
    def detect_with_landmarks(img: t.RGBImg) -> Tuple[t.Scores, t.Bboxes, t.Landmarks, t.Additional]:
        img, scale,true_scale = prepare_img(img)
        scores, bboxes, landmarks, angles = rt_sess.run(None, {input_name: img})

        scor = []
        bbox = []
        face_points = []
        angle = []

        if len(scores) > 0:
            # print(bboxes)
            # print(bboxes*true_scale,true_scale)

            bboxes *= scale
            # print(bboxes, scale)
            landmarks *= scale
            landmarks = np.reshape(landmarks, (bboxes.shape[0], -1, 2))

        for i in range(len(scores)):
            if scores[i]>threshold:
                scor.append(scores[i])
                angle.append(angles[i])

                left_eye = np.mean(landmarks[i][36:42],axis=0)
                right_eye = np.mean(landmarks[i][42:48],axis=0)
                nose = np.mean(landmarks[i][30:31],axis=0)
                left_mouth = np.mean(landmarks[i][48:49],axis=0)
                right_mouth = np.mean(landmarks[i][54:55],axis=0)  
                face_points.append(np.stack((left_eye,right_eye,nose,left_mouth,right_mouth)).astype(int).T.flatten())

                x,y,x1,y1 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
                center_x = (x1+x)//2
                center_y = (y1+y)//2
                half_border = max(x1-x,y1-y)//2 
                x = center_x - half_border
                x1 = center_x + half_border
                y = center_y - half_border
                y1 = center_y + half_border 
                bbox.append(np.array([x,y,x1,y1],dtype=int))

        return np.array(scor), np.array(bbox), np.array(face_points), np.array(angle), np.array(landmarks)
    
    if lands:
        return detect_with_landmarks
    else:
        return detect