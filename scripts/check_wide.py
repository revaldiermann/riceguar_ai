
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import math

DATA_DIR = Path("dataset/split/train")
IMAGE_SIZE = 224
MAX_IMAGES_PER_CLASS = 4
MIN_ASPECT_RATIO = 2.0

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Folder tidak ditemukan: {DATA_DIR}")

extensions = {".jpg", ".jpeg", ".png", ".webp"}
classes = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())

if not classes:
    raise RuntimeError("Tidak ditemukan folder kelas.")

samples = []

for class_dir in classes:
    wide_images = []

    for path in sorted(class_dir.iterdir()):
        if path.suffix.lower() not in extensions:
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size

                if width / height >= MIN_ASPECT_RATIO:
                    wide_images.append(path)
        except Exception as exc:
            print(f"Gagal membaca {path}: {exc}")

    selected = wide_images[:MAX_IMAGES_PER_CLASS]

    print(
        f"{class_dir.name}: "
        f"{len(wide_images)} gambar lebar ditemukan, "
        f"{len(selected)} ditampilkan"
    )

    samples.extend((class_dir.name, path) for path in selected)

if not samples:
    raise RuntimeError("Tidak ditemukan gambar dengan rasio minimal 2:1.")

fig, axes = plt.subplots(
    len(samples),
    3,
    figsize=(12, max(3, len(samples) * 2.5)),
    squeeze=False,
)

for row, (class_name, path) in enumerate(samples):
    with Image.open(path) as img:
        original = img.convert("RGB")

    stretched = original.resize((IMAGE_SIZE, IMAGE_SIZE))

    contained = ImageOps.contain(
        original,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    padded = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        (0, 0, 0)
    )

    padded.paste(
        contained,
        (
            (IMAGE_SIZE - contained.width) // 2,
            (IMAGE_SIZE - contained.height) // 2,
        ),
    )

    axes[row, 0].imshow(original)
    axes[row, 0].set_title(
        f"{class_name}\nAsli: {original.width}×{original.height}"
    )

    axes[row, 1].imshow(stretched)
    axes[row, 1].set_title("Resize langsung")

    axes[row, 2].imshow(padded)
    axes[row, 2].set_title("Rasio + padding")

    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()

output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "wide_image_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nVisualisasi disimpan di: {output_path}")

plt.show()
