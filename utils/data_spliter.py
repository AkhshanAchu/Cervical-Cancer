import random
from collections import defaultdict

def train_val_split(samples, val_ratio=0.2, seed=42):
    random.seed(seed)
    image_groups = defaultdict(list)
    for s in samples:
        image_groups[s[0]].append(s)

    images = list(image_groups.keys())
    random.shuffle(images)

    split_idx = int(len(images) * (1 - val_ratio))
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    train_samples = []
    val_samples = []

    for img in train_imgs:
        train_samples.extend(image_groups[img])
    for img in val_imgs:
        val_samples.extend(image_groups[img])

    print(f"Train images: {len(train_imgs)}")
    print(f"Val images: {len(val_imgs)}")
    print(f"Train cells: {len(train_samples)}")
    print(f"Val cells: {len(val_samples)}")

    return train_samples, val_samples
