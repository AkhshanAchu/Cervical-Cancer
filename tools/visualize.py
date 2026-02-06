import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

def visualize_sample(img, cyt, nuc, idx=0):
    """
    img : Tensor [B, 3, H, W]
    cyt : Tensor [B, 1, H, W]
    nuc : Tensor [B, 1, H, W]
    idx : which sample in batch to visualize
    """

    img = img[idx].permute(1, 2, 0).cpu().numpy()
    cyt = cyt[idx, 0].cpu().numpy()
    nuc = nuc[idx, 0].cpu().numpy()

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(img)
    axs[0].set_title("RGB Image")
    axs[0].axis("off")

    axs[1].imshow(cyt, cmap="gray")
    axs[1].set_title("Cytoplasm Mask")
    axs[1].axis("off")

    axs[2].imshow(nuc, cmap="gray")
    axs[2].set_title("Nucleus Mask")
    axs[2].axis("off")

    # Overlay (very useful)
    overlay = img.copy()
    overlay[..., 1] = np.maximum(overlay[..., 1], cyt * 0.7)  # green = cyt
    overlay[..., 0] = np.maximum(overlay[..., 0], nuc * 0.9)  # red = nuc

    axs[3].imshow(overlay)
    axs[3].set_title("Overlay (Green=Cyt, Red=Nuc)")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

@torch.no_grad()
def sliding_window_inference(
    model,
    image,
    patch_size=256,
    stride=128,
    device = "cuda"
):
    """
    image: numpy array [H, W, 3]
    returns:
        cyt_mask, nuc_mask -> [H, W] probability maps
    """

    H, W, _ = image.shape

    cyt_pred = np.zeros((H, W), dtype=np.float32)
    nuc_pred = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]

            patch = torch.from_numpy(patch)\
                         .permute(2, 0, 1)\
                         .unsqueeze(0)\
                         .to(device)

            out = model(patch)

            cyt = torch.sigmoid(out["cyt"])[0, 0].cpu().numpy()
            nuc = torch.sigmoid(out["nuc"])[0, 0].cpu().numpy()

            cyt_pred[y:y+patch_size, x:x+patch_size] += cyt
            nuc_pred[y:y+patch_size, x:x+patch_size] += nuc
            count_map[y:y+patch_size, x:x+patch_size] += 1

    # Avoid division by zero
    count_map[count_map == 0] = 1

    cyt_pred /= count_map
    nuc_pred /= count_map

    return cyt_pred, nuc_pred

def load_cropped_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

@torch.no_grad()
def infer_single_image(model, img,device='cuda'):
    """
    img: numpy array [256, 256, 3]
    """
    x = torch.from_numpy(img)\
             .permute(2, 0, 1)\
             .unsqueeze(0)\
             .to(device)

    out = model(x)

    cyt_prob = torch.sigmoid(out["cyt"])[0, 0].cpu().numpy()
    nuc_prob = torch.sigmoid(out["nuc"])[0, 0].cpu().numpy()

    return cyt_prob, nuc_prob

def plot_segemnt(image_path, model):
    img = load_cropped_image(image_path)
    cyt_prob, nuc_prob = infer_single_image(model, img)
    cyt_mask = (cyt_prob > 0.5).astype(np.uint8)
    nuc_mask = (nuc_prob > 0.5).astype(np.uint8)
    nuc_mask = nuc_mask * cyt_mask
    h, w = img.shape[:2]

    cyt_mask_vis = cv2.resize(
        cyt_mask, (w, h), interpolation=cv2.INTER_NEAREST
    )

    nuc_mask_vis = cv2.resize(
        nuc_mask, (w, h), interpolation=cv2.INTER_NEAREST
    )

    overlay = img.copy()
    overlay[..., 1] = np.maximum(overlay[..., 1], cyt_mask_vis * 0.7)
    overlay[..., 0] = np.maximum(overlay[..., 0], nuc_mask_vis * 0.9)

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.title("Overlay (Green=Cyt, Red=Nuc)")
    plt.axis("off")
    plt.show()

