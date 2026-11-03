import os
from src.data.dataloader import get_dataloaders


def main():
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))

    # Menyusun teks laporan
    report_content = ""
    report_content += "=" * 45 + "\n"
    report_content += "RICEGUARD AI - DATA LOADER TEST\n"
    report_content += "=" * 45 + "\n"

    report_content += "\nJumlah gambar:\n"
    report_content += f"Train      : {len(train_loader.dataset)}\n"
    report_content += f"Validation : {len(val_loader.dataset)}\n"
    report_content += f"Test       : {len(test_loader.dataset)}\n"

    report_content += "\nNama kelas:\n"
    report_content += f"{train_loader.dataset.classes}\n"

    report_content += "\nPemetaan label:\n"
    report_content += f"{train_loader.dataset.class_to_idx}\n"

    report_content += "\nBatch pertama:\n"
    report_content += f"Shape gambar : {images.shape}\n"
    report_content += f"Shape label  : {labels.shape}\n"
    report_content += f"Tipe gambar  : {images.dtype}\n"
    report_content += f"Label        : {labels.tolist()}\n"

    report_content += "\nJumlah batch:\n"
    report_content += f"Train      : {len(train_loader)}\n"
    report_content += f"Validation : {len(val_loader)}\n"
    report_content += f"Test       : {len(test_loader)}\n"

    report_content += "\nDataLoader berhasil diuji.\n"

    # 1. Tampilkan hasil pengujian ke layar terminal
    print(report_content)

    # 2. Simpan hasil ke dalam folder reports/
    # Membuat folder reports jika belum ada
    os.makedirs("reports", exist_ok=True)
    
    # Menulis laporan ke file teks
    path_report = os.path.join("reports", "test_dataloader_report.txt")
    with open(path_report, "w") as f:
        f.write(report_content)
        
    print(f"Hasil pengujian DataLoader telah dicatat di: {path_report}")


if __name__ == "__main__":
    main()
