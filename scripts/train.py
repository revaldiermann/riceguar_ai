import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.data.dataloader import get_dataloaders
from src.models.efficientnet import build_model


EPOCHS = 15
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = Path("checkpoints/best_model.pth")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


def main():
    torch.manual_seed(42)

    device = get_device()
    
    # Inisialisasi string laporan
    report_content = "=" * 45 + "\n"
    report_content += "RICEGUARD AI - TRAINING REPORT\n"
    report_content += "=" * 45 + "\n"
    
    log_device = f"Device: {device}\n"
    print(log_device, end="")
    report_content += log_device

    train_loader, val_loader, _ = get_dataloaders()

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device
        )

        # Menyusun baris log untuk epoch ini
        epoch_log = (
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%}\n"
        )
        print(epoch_log, end="")
        report_content += epoch_log

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                CHECKPOINT_PATH,
            )

            save_log = f"  -> Model terbaik disimpan: {CHECKPOINT_PATH}\n"
            print(save_log, end="")
            report_content += save_log

    # Menyusun penutup log setelah training selesai
    finish_log = "\nTraining selesai.\n"
    finish_log += f"Best validation loss: {round(best_val_loss, 4)}\n"
    finish_log += f"Checkpoint: {CHECKPOINT_PATH}\n"
    
    print(finish_log)
    report_content += finish_log

    # Menulis seluruh riwayat ke file teks di folder reports/
    os.makedirs("reports", exist_ok=True)
    path_report = os.path.join("reports", "train_report.txt")
    with open(path_report, "w") as f:
        f.write(report_content)
        
    print(f"Hasil rekaman training telah dicatat di: {path_report}")


if __name__ == "__main__":
    main()
