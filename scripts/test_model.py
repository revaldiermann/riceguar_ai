import os
import torch
from src.models.efficientnet import build_model


def main():
    model = build_model(pretrained=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    # Menyiapkan teks hasil output
    hasil_output = (
        f"Model: {model.__class__.__name__}\n"
        f"Input shape: {dummy_input.shape}\n"
        f"Output shape: {output.shape}\n"
        f"Jumlah kelas: {output.shape[1]}\n"
    )

    # 1. Tampilkan hasil di terminal seperti biasa
    print(hasil_output, end="")

    assert output.shape == (1, 3)
    sukses_msg = "Model berhasil diuji.\n"
    print(sukses_msg)
    hasil_output += sukses_msg

    # 2. Simpan hasil ke dalam folder reports/
    # Membuat folder reports jika belum ada
    os.makedirs("reports", exist_ok=True)
    
    # Menulis hasil ke file teks di folder reports
    path_report = os.path.join("reports", "test_model_report.txt")
    with open(path_report, "w") as f:
        f.write(hasil_output)
        
    print(f"Hasil pengujian telah dicatat di: {path_report}")


if __name__ == "__main__":
    main()
