import matplotlib.pyplot as plt
import numpy as np

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
