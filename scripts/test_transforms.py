import os
from pathlib import Path
from PIL import Image

from src.data.transforms import (
    train_transforms,
    val_test_transforms,
)


def main():
    DATA_DIR = Path("dataset/split/train")
    image_paths = list(DATA_DIR.glob("*/*.jpg"))

    if not image_paths:
        image_paths = list(DATA_DIR.glob("*/*.jpeg"))

    if not image_paths:
        image_paths = list(DATA_DIR.glob("*/*.png"))

    if not image_paths:
        raise RuntimeError("Tidak ditemukan gambar di dataset/split/train")

    image_path = image_paths[0]

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        train_tensor = train_transforms(image)
        val_tensor = val_test_transforms(image)

    # Menyusun teks laporan secara dinamis berdasarkan hasil eksekusi gambar asli
    report_content = ""
    report_content += "=" * 45 + "\n"
    report_content += "RICEGUARD AI - TRANSFORMS TEST\n"
    report_content += "=" * 45 + "\n"
    report_content += f"File gambar: {image_path}\n"
    report_content += f"Shape train: {train_tensor.shape}\n"
    report_content += f"Shape validation: {val_tensor.shape}\n"
    report_content += f"Tipe data: {train_tensor.dtype}\n"
    report_content += f"Nilai minimum: {train_tensor.min().item()}\n"
    report_content += f"Nilai maksimum: {train_tensor.max().item()}\n"
    report_content += "\nTransformasi data berhasil diuji.\n"

    # 1. Tampilkan hasil di layar terminal
    print(report_content)

    # 2. Membuat folder reports jika belum ada dan menyimpan hasilnya ke file teks
    os.makedirs("reports", exist_ok=True)
    path_report = os.path.join("reports", "test_transforms_report.txt")
    with open(path_report, "w") as f:
        f.write(report_content)

    print(f"Hasil pengujian Transforms telah dicatat di: {path_report}")


if __name__ == "__main__":
    main()
