import cv2
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from pathlib import Path
import torch

def load_polygon(dat_path):
    pts = []
    with open(dat_path, "r") as f:
        for line in f:
            x, y = line.strip().split(",")
            pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.int32)


def polygon_to_mask(poly, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    return mask

def crop_from_polygon(img, cyt_mask, nuc_mask, poly, margin=32):
    h, w = img.shape[:2]

    x, y, bw, bh = cv2.boundingRect(poly)

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)

    img_c = img[y1:y2, x1:x2]
    cyt_c = cyt_mask[y1:y2, x1:x2]
    nuc_c = nuc_mask[y1:y2, x1:x2]

    return img_c, cyt_c, nuc_c

def collect_all_samples(root_dir):
    root = Path(root_dir)
    samples = []

    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue

        for img_path in class_dir.glob("*.bmp"):
            stem = img_path.stem
            cyt_files = sorted(class_dir.glob(f"{stem}_cyt*.dat"))

            for cyt_path in cyt_files:
                idx = cyt_path.stem.split("cyt")[-1]
                nuc_path = class_dir / f"{stem}_nuc{idx}.dat"

                if nuc_path.exists():
                    samples.append((img_path, cyt_path, nuc_path))

    print(f"Total cell instances: {len(samples)}")
    return samples
