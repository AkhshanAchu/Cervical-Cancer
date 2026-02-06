import torch
import torch.nn as nn
from  tqdm import tqdm


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    lr=1e-4,
    device="cuda",
    save_path="best_model_mutliclass.pth",
    criterion = None
):
    model.to(device)
    if not criterion:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 30)

        # ================= TRAIN =================
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc="Training", leave=False)

        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # stats
            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += batch_size

            # tqdm update
            train_pbar.set_postfix({
                "loss": f"{train_loss / total:.4f}",
                "acc": f"{correct / total:.4f}"
            })

        train_loss /= total
        train_acc = correct / total

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        val_pbar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += batch_size

                # tqdm update
                val_pbar.set_postfix({
                    "loss": f"{val_loss / total:.4f}",
                    "acc": f"{correct / total:.4f}"
                })

        val_loss /= total
        val_acc = correct / total
        scheduler.step(val_loss)

        # ================= LOGGING =================
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # ================= SAVE BEST =================
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), save_path)
            best_val_acc = val_acc
            print("✅ Saved best model")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
