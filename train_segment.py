from tqdm import tqdm
from utils.loss import segmentation_loss,dice_loss,dice_score
import torch
import os

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)

    for img, cyt, nuc in pbar:
        img = img.to(device)
        cyt = cyt.to(device)
        nuc = nuc.to(device)

        optimizer.zero_grad()
        out = model(img)
        loss = segmentation_loss(out, cyt, nuc)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    loss_sum = 0.0
    dice_cyt_sum = 0.0
    dice_nuc_sum = 0.0

    pbar = tqdm(loader, desc="Val  ", leave=False)

    for img, cyt, nuc in pbar:
        img = img.to(device)
        cyt = cyt.to(device)
        nuc = nuc.to(device)

        out = model(img)
        loss = segmentation_loss(out, cyt, nuc)

        loss_sum += loss.item()
        dice_cyt_sum += dice_score(out["cyt"], cyt)
        dice_nuc_sum += dice_score(out["nuc"], nuc)

    return (
        loss_sum / len(loader),
        dice_cyt_sum / len(loader),
        dice_nuc_sum / len(loader),
    )


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=150,
    save_dir="checkpoints",
    save_every=10,
):
    os.makedirs(save_dir, exist_ok=True)

    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch [{epoch}/{epochs}]")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device
        )

        val_loss, dice_cyt, dice_nuc = validate(
            model, val_loader, device
        )

        mean_dice = (dice_cyt + dice_nuc) / 2

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Cyt Dice: {dice_cyt:.4f} | "
            f"Nuc Dice: {dice_nuc:.4f}"
        )

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_model.pth")
            )
            print(f"Saved BEST model (Mean Dice = {best_dice:.4f})")

        if epoch % save_every == 0:
            ckpt_path = os.path.join(
                save_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
