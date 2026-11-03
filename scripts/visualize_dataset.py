
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path("dataset/train")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(42)

class_dirs = sorted(
    folder for folder in DATA_DIR.iterdir()
    if folder.is_dir() and not folder.name.startswith(".")
)

if not class_dirs:
    raise RuntimeError(f"Tidak ada folder kelas di {DATA_DIR}")

fig, axes = plt.subplots(
    nrows=len(class_dirs),
    ncols=3,
    figsize=(12, 4 * len(class_dirs))
)

for row, class_dir in enumerate(class_dirs):
    images = [
        path for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        for ax in axes[row]:
            ax.axis("off")
        continue

    samples = random.sample(images, min(3, len(images)))

    for col, ax in enumerate(axes[row]):
        ax.axis("off")

        if col < len(samples):
            with Image.open(samples[col]) as img:
                ax.imshow(img.convert("RGB"))

            ax.set_title(
                f"{class_dir.name}\n{samples[col].name}",
                fontsize=9
            )

plt.tight_layout()

output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "dataset_samples.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Visualisasi tersimpan di: {output_path}")
