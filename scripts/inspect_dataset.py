
from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError

# Sesuaikan jika folder kelas berada di lokasi lain.
DATA_DIR = Path("dataset/train")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

class_counts = Counter()
extensions = Counter()
image_sizes = Counter()
corrupted_files = []
total_images = 0

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Folder {DATA_DIR} tidak ditemukan. "
        "Periksa lokasi dataset."
    )

class_dirs = sorted(
    folder for folder in DATA_DIR.iterdir()
    if folder.is_dir() and not folder.name.startswith(".")
)

if not class_dirs:
    raise RuntimeError(
        "Tidak ditemukan folder kelas langsung di dalam data/. "
        "Periksa apakah dataset berada di dalam data/train/."
    )

print("=" * 55)
print("RICEGUARD AI - DATASET INSPECTION")
print("=" * 55)

for class_dir in class_dirs:
    class_name = class_dir.name

    files = [
        path for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    class_counts[class_name] = len(files)
    total_images += len(files)

    print(f"\nKelas: {class_name}")
    print(f"Jumlah file gambar: {len(files)}")

    for path in files:
        extensions[path.suffix.lower()] += 1

        try:
            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                image_sizes[img.size] += 1

        except (OSError, ValueError, UnidentifiedImageError):
            corrupted_files.append(str(path))

print("\n" + "=" * 55)
print("RINGKASAN DATASET")
print("=" * 55)
print(f"Total gambar : {total_images}")
print(f"Total kelas  : {len(class_counts)}")

print("\nDistribusi kelas:")
for name, count in class_counts.items():
    percentage = count / total_images * 100 if total_images else 0
    print(f"- {name}: {count} gambar ({percentage:.2f}%)")

print("\nFormat gambar:")
for ext, count in extensions.most_common():
    print(f"- {ext}: {count}")

print("\n10 ukuran gambar yang paling sering muncul:")
for size, count in image_sizes.most_common(10):
    print(f"- {size[0]} x {size[1]}: {count} gambar")

print(f"\nGambar rusak: {len(corrupted_files)}")
for path in corrupted_files[:10]:
    print(f"- {path}")

print("\nPemeriksaan selesai.")
