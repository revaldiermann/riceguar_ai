
import argparse
from pathlib import Path

import torch
from PIL import Image

from src.data.transforms import val_test_transforms
from src.models.efficientnet import build_model


# Urutan kelas harus sama dengan urutan ImageFolder saat training.
CLASSES = [
    "Bacterial leaf blight",
    "Brown spot",
    "Leaf smut",
]

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")


def main():
    parser = argparse.ArgumentParser(
        description="Prediksi penyakit daun padi menggunakan RiceGuard AI"
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Lokasi file gambar daun padi yang akan diprediksi",
    )
    args = parser.parse_args()

    # Periksa file gambar dan checkpoint.
    if not args.image.is_file():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {args.image}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {CHECKPOINT_PATH}"
        )

    # Pilih perangkat komputasi.
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Muat model dan bobot hasil training.
    model = build_model(pretrained=False).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Baca gambar dan gunakan preprocessing yang sama dengan evaluasi.
    with Image.open(args.image) as img:
        image = img.convert("RGB")

    image_tensor = val_test_transforms(image).unsqueeze(0).to(device)

    # Jalankan prediksi.
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = probabilities.argmax().item()
    confidence = probabilities[predicted_index].item() * 100

    print("\n" + "=" * 45)
    print("          HASIL PREDIKSI RICEGUARD AI")
    print("=" * 45)
    print(f"Gambar       : {args.image}")
    print(f"Hasil        : {CLASSES[predicted_index]}")
    print(f"Confidence   : {confidence:.2f}%")

    print("\nProbabilitas setiap kelas:")
    for index in probabilities.argsort(descending=True).tolist():
        probability = probabilities[index].item() * 100
        print(f"- {CLASSES[index]}: {probability:.2f}%")

    print("=" * 45)


if __name__ == "__main__":
    main()
