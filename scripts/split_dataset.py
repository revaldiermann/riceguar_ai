from pathlib import Path
from collections import Counter
import random
import shutil
import hashlib

SOURCE_DIR = Path("dataset/train")
OUTPUT_DIR = Path("dataset/split")

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = { ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff" }
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Folder tidak ditemukan: {SOURCE_DIR}")

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise RuntimeError(
        f"{OUTPUT_DIR} sudah berisi data. "
        "Hentikan skrip agar file lama tidak tertimpa. "
        "Periksa folder tersebut sebelum mencoba lagi."
    )

random.seed(SEED)

class_dirs = sorted(
    p for p in SOURCE_DIR.iterdir()
    if p.is_dir() and not p.name.startswith(".")
)

if len(class_dirs) != 3:
    raise RuntimeError(
        f"Diharapkan 3 kelas, tetapi ditemukan {len(class_dirs)}."
    )

# Kumpulkan file per kelas
files_by_class = {}

for class_dir in class_dirs:
    files = sorted(
        p for p in class_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(files) < 3:
        raise RuntimeError(f"Gambar terlalu sedikit di {class_dir.name}")

    files_by_class[class_dir.name] = files

# Deteksi duplikat persis berdasarkan isi file
hash_to_paths = {}

for class_name, files in files_by_class.items():
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_to_paths.setdefault(digest, []).append((class_name, path))

duplicates = [
    paths for paths in hash_to_paths.values()
    if len(paths) > 1
]

if duplicates:
    print("PERINGATAN: ditemukan duplikat persis.")
    for group in duplicates:
        print(" - " + ", ".join(str(path) for _, path in group))

    raise RuntimeError(
        "Pembagian dibatalkan. Periksa duplikat di atas terlebih dahulu."
    )

print("Tidak ditemukan duplikat persis.")

# Bagi secara acak per kelas
splits = {"train": {}, "val": {}, "test": {}}

for class_name, files in files_by_class.items():
    files = files.copy()
    random.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits["train"][class_name] = files[:n_train]
    splits["val"][class_name] = files[n_train:n_train + n_val]
    splits["test"][class_name] = files[n_train + n_val:]

# Salin file ke struktur output
for split_name, classes in splits.items():
    for class_name, files in classes.items():
        target_dir = OUTPUT_DIR / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for source_path in files:
            shutil.copy2(source_path, target_dir / source_path.name)

# Tampilkan hasil
print("\n=== HASIL PEMBAGIAN DATASET ===")

for split_name, classes in splits.items():
    print(f"\n{split_name.upper()}")

    for class_name, files in classes.items():
        print(f"  {class_name}: {len(files)} gambar")

    print(f"  Total: {sum(len(v) for v in classes.values())}")

print(f"\nDataset hasil pembagian tersimpan di: {OUTPUT_DIR}")
print("Dataset asli tidak diubah.")


