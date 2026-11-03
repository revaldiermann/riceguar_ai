
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

image_path = Path(
    "dataset/split/train/Bacterial leaf blight/DSC_0700.jpg"
)

with Image.open(image_path) as image:
    original = image.convert("RGB")

    # A: Resize langsung; rasio gambar berubah
    stretched = original.resize((224, 224))

    # B: Pertahankan rasio, lalu tambahkan padding
    contained = ImageOps.contain(original, (224, 224))
    padded = Image.new("RGB", (224, 224), (0, 0, 0))
    padded.paste(
        contained,
        (
            (224 - contained.width) // 2,
            (224 - contained.height) // 2,
        ),
    )

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(original)
axes[0].set_title(f"Asli: {original.size}")

axes[1].imshow(stretched)
axes[1].set_title("Resize langsung")

axes[2].imshow(padded)
axes[2].set_title("Rasio dipertahankan")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
