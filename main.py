from utils.data_spliter import train_val_split
from utils.dataloader import CellSegDataset
from tools.helper import collect_all_samples
from torch.utils.data import DataLoader
from tools.visualize import visualize_sample
import torch
from model.model_seg import DualHeadAttConvNeXtUNet
from train import train_model


root = r"C:\Users\akhsh\Desktop\Cancer\Data"

all_samples = collect_all_samples(root)

train_samples, val_samples = train_val_split(
    all_samples,
    val_ratio=0.2
)

train_ds = CellSegDataset(train_samples, out_size=256)
val_ds = CellSegDataset(val_samples, out_size=256)

train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

img, cyt, nuc = next(iter(train_loader))

print(img.shape)
print(cyt.shape)
print(nuc.shape)

# visualize_sample(img, cyt, nuc, idx=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training in : ",device)
model = DualHeadAttConvNeXtUNet().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

train_model(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    epochs=50,
    save_dir="cellseg_checkpoints_2",
    save_every=10
)

print("Training Done...")