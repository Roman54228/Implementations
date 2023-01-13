import numpy as np
from typing import Tuple, Callable, Sequence


Img = np.ndarray  # any 3d array
BGRImg = np.ndarray  # Img array with channels in Blue Green Red order (after cv.imread)
RGBImg = np.ndarray  # Img array with channels in Red Green Blue order (for the input)
Scores = np.ndarray  # vector of detection scores of length N, where N is the number of detections
Bboxes = np.ndarray  # Nx4 matrix of detected bounding boxes in (x_min, y_min, x_max, y_max) format
Landmarks = np.ndarray  # NxLx2 tensor of detected landmarks keypoints, where L is the number of landmarks (5 or 68)
Additional = np.ndarray  # NxM matrix of additional information of detected object

Detector = Callable[[RGBImg], Tuple[Scores, Bboxes, Landmarks, Additional]]
Processor = Callable[[BGRImg], BGRImg]
PartProcessor = Callable[[RGBImg], Sequence[np.ndarray]]
