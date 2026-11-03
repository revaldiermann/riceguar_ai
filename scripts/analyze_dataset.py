from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from setuptools import Extension

DATA_DIR = Path("dataset/train")
IMAGE_EXTENSIONS = {
  ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

records = []
corrupted = []
sizes = Counter()
extension = Counter()

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Folder {DATA_DIR} tidak ditemukan. "
        "Periksa lokasi dataset."
    )

class_dirs = sorted(
  p for p in DATA_DIR.iterdir()
  if p.is_dir() and not p.name.startswith(".")
)
for class_dir in class_dirs:
    for path in class_dir.rglob("*"):
         if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
         
         extension[path.suffix.lower()] += 1
         try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode

            sizes[(width, height)] += 1

            records.append({
                "filename": path.name,
                "filepath": str(path),
                "class": class_dir.name,
                "width": width,
                "height": height,
                "mode": mode,
                "format": path.suffix.lower(),
            })
         except Exception as error:
            corrupted.append({
                "filepath": str(path),
                "error": str(error),
            })

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("Tidak ada gambar valid yang ditemukan.")

print("\n=== JUMLAH GAMBAR PER KELAS ===")
counts = df["class"].value_counts().sort_index()
print(counts.to_string())

print("\n=== DISTRIBUSI PERSENTASE ===")
print((counts / counts.sum() * 100).round(2).astype(str).add("%"))

print("\n=== UKURAN GAMBAR PALING UMUM ===")
print(df.groupby(["width", "height"]).size()
      .sort_values(ascending=False).head(10))

print("\n=== MODE WARNA ===")
print(df["mode"].value_counts().to_string())

print("\n=== FORMAT GAMBAR ===")
print(df["format"].value_counts().to_string())

print(f"\nGambar valid: {len(df)}")
print(f"Gambar rusak: {len(corrupted)}")

# Simpan laporan
report_dir = Path("reports")
figure_dir = report_dir / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(report_dir / "dataset_inventory.csv", index=False)

if corrupted:
    pd.DataFrame(corrupted).to_csv(
        report_dir / "corrupted_images.csv", index=False
    )

# Grafik distribusi kelas
ax = counts.plot(kind="bar", figsize=(9, 5))
ax.set_title("RiceGuard AI - Image Distribution by Class")
ax.set_xlabel("Class")
ax.set_ylabel("Number of Images")
plt.xticks(rotation=0, ha="right")
plt.tight_layout()
plt.savefig(figure_dir / "class_distribution.png", dpi=150)
plt.show()

print("\nLaporan tersimpan di folder reports/.")
        

