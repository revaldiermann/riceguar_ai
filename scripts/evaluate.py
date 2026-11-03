
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.data.dataloader import get_dataloaders
from src.models.efficientnet import build_model


CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {CHECKPOINT_PATH}"
        )

    device = get_device()
    _, _, test_loader = get_dataloaders()
    class_names = test_loader.dataset.classes

    # Tidak perlu mengunduh bobot pretrained lagi.
    model = build_model(pretrained=False).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()

            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    accuracy = sum(
        true == pred for true, pred in zip(y_true, y_pred)
    ) / len(y_true)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("=" * 50)
    print("RICEGUARD AI - TEST EVALUATION")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Test images: {len(y_true)}")
    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(report)

    report_path = REPORT_DIR / "test_classification_report.txt"
    with report_path.open("w", encoding="utf-8") as file:
        file.write(f"Test accuracy: {accuracy:.4%}\n\n")
        file.write(report)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    display.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=25)
    ax.set_title("RiceGuard AI - Test Confusion Matrix")
    fig.tight_layout()

    figure_path = REPORT_DIR / "figures" / "test_confusion_matrix.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Laporan disimpan:", report_path)
    print("Confusion matrix disimpan:", figure_path)


if __name__ == "__main__":
    main()
