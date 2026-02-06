import os
import random
from pathlib import Path
import numpy as np
import cv2

def load_polygon(dat_path):
    pts = []
    with open(dat_path, "r") as f:
        for line in f:
            x, y = line.strip().split(",")
            pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.float32)


def rasterize_union(polygons, shape):
    mask = np.zeros(shape, np.uint8)
    for p in polygons:
        cv2.fillPoly(mask, [p.astype(np.int32)], 1)
    return mask
