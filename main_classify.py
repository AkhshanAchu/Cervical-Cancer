from utils.dataloader import CellClassificationDataset
import random
import torch
from torch.utils.data import DataLoader
import os
from model.model_classifier import ConvNeXtAttentionClassifier
from train_classify import train_model
from evaluate import evaluate_model_classification, plot_confusion_matrix

dataset = CellClassificationDataset(
    root_dir=r"C:\Users\akhsh\Desktop\Cancer\Data"
)

n = len(dataset)
idx = list(range(n))
random.shuffle(idx)

split = int(0.8 * n)
train_idx, val_idx = idx[:split], idx[split:]

train_ds = torch.utils.data.Subset(dataset, train_idx)
val_ds   = torch.utils.data.Subset(dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)

save_dir = "checkpoint_classification_2"
print("Save dir : ",save_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ConvNeXtAttentionClassifier(
    num_classes=5,
    input_channels=5
).to(device)

os.makedirs(save_dir,exist_ok=True)

train_model(
model,
train_loader=train_loader,
val_loader=val_loader,
num_epochs=20,
lr=1e-4,
device="cuda",
save_path=save_dir+"/best_model_mutliclass.pth"
)

print("Training Doneeee...")

class_names = list({'Dyskeratotic': 0, 'Koilocytotic': 1, 'Metaplastic': 2, 'Parabasal': 3, 'Superficial-Intermediate': 4}.keys())

cm, y_true, y_pred = evaluate_model_classification(
    model,
    dataloader=val_loader,
    device=device,
    class_names=class_names,
    checkpoint_path="checkpoint_classification/best_model_mutliclass.pth"
)

plot_confusion_matrix(cm, class_names)
