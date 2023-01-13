from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as rt

import mytypes as t


def prepare_session(model_path: Path) -> rt.InferenceSession:
    return rt.InferenceSession(str(model_path))


def get_detector(model_path: Path, detector_shape: int) -> t.Detector:
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
        return new_img, max(width, height)

    def detect(img: t.RGBImg) -> Tuple[t.Scores, t.Bboxes, t.Landmarks, t.Additional]:
        img, scale = prepare_img(img)
        scores, bboxes, landmarks, angles = rt_sess.run(None, {input_name: img})
        if len(scores) > 0:
            bboxes *= scale
            landmarks *= scale
            landmarks = np.reshape(landmarks, (bboxes.shape[0], -1, 2))
        else:
            landmarks = np.empty((0, 68, 2), dtype=np.float32)
        return scores, bboxes, landmarks, angles

    return detect
