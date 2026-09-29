# Prediksi Prioritas *Follow-up Repeat Order* — SPARC 2026

Proyek *machine learning* end-to-end untuk memprediksi pelanggan mana yang berpotensi melakukan
**pembelian ulang (*repeat order*)**, sehingga tim penjualan dapat **memprioritaskan aktivitas *follow-up***
pada pelanggan dengan peluang repeat tertinggi. Dikerjakan untuk kompetisi **SPARC 2026** oleh **Tim TrioTunggal**.

> **TL;DR** — Dari data transaksi penjualan kendaraan (tanpa label), dibangun target repeat di level pelanggan,
> lalu dilatih model **LightGBM** yang menghasilkan **ROC-AUC ≈ 0.75**. Model mengubah keputusan *follow-up* dari
> acak menjadi terarah: **menghubungi 30% pelanggan teratas (skor tertinggi) sudah menjangkau ±60% dari seluruh
> repeater**, dan desil teratas memiliki **lift ≈ 2.7×** di atas rata-rata.

---

## 1. Latar Belakang & Rumusan Masalah

Sebuah perusahaan otomotif ingin memanfaatkan data transaksinya untuk **menentukan prioritas *follow-up*
repeat order**. Tantangannya: dataset bersifat **transaksional** (satu baris = satu transaksi) dan **tidak
memiliki kolom target** yang siap pakai.

**Pertanyaan bisnis:** *Berdasarkan profil transaksi pertama seorang pelanggan, seberapa besar peluang ia akan
melakukan pembelian ulang — dan pelanggan mana yang harus dihubungi lebih dulu?*

## 2. Pendekatan Solusi

| Tahap | Ringkasan |
|------|-----------|
| **Target engineering** | `is_repeat = 1` bila pelanggan punya > 1 transaksi. Dibentuk **sebelum** pembersihan fitur agar label tidak terkontaminasi. |
| **Anti-leakage** | Tiap pelanggan diwakili **transaksi pertamanya** (paling awal), bukan terakhir — meniru kondisi nyata saat *follow-up* diputuskan. |
| **Pembersihan data** | Parsing nilai Rupiah & tanggal (format campur), koreksi umur dari tanggal lahir, penanganan anomali berbasis logika bisnis. |
| **Feature engineering** | Transformasi log (OTR/DP/cicilan), fitur rasio (`dp_ratio`, `cicilan_ratio`), *frequency encoding* untuk kolom *high-cardinality*. |
| **Encoding & pipeline** | `ColumnTransformer` (imputasi + scaling + One-Hot) di dalam `Pipeline` scikit-learn — mencegah kebocoran saat cross-validation. |
| **Pemodelan** | Baseline (Dummy) → Logistic Regression → Random Forest → **LightGBM** (juara), dengan penanganan *class imbalance* & *hyperparameter tuning*. |
| **Evaluasi** | ROC-AUC, PR-AUC, Precision/Recall/F1, *threshold tuning*, *feature importance* (gain + permutation). |
| **Output bisnis** | Tabel *lift*/*gains* + `prioritas_followup.csv` berisi skor & tier prioritas tiap pelanggan. |

## 3. Dataset

- **Sumber:** disediakan panitia SPARC 2026 (± 320.000 transaksi, 28 kolom).
- **Data import di notebook hanya melalui tautan GitHub resmi kompetisi**, sesuai ketentuan lomba.
- Kelas target **tidak seimbang**: hanya ± **13%** pelanggan yang melakukan repeat order.

> ⚠️ **Penting (aturan kompetisi):** dataset **dilarang didistribusikan/dipublikasikan**. File `SPARC_dataset.csv`
> dan seluruh keluaran turunan (mis. `prioritas_followup.csv`) **sengaja tidak diikutkan** ke repositori

## 4. Hasil Utama

Perbandingan model pada data uji (20% *holdout*, *stratified*):

| Model | ROC-AUC | PR-AUC | F1 | Recall |
|-------|:------:|:-----:|:--:|:-----:|
| Dummy (baseline) | 0.50 | 0.13 | 0.13 | 0.13 |
| Logistic Regression | ~0.71 | ~0.25 | ~0.33 | ~0.67 |
| Random Forest | ~0.73 | ~0.27 | ~0.34 | ~0.46 |
| **LightGBM (tuned)** | **~0.75** | **~0.33** | **~0.36** | **~0.71** |

**Dampak bisnis (Cumulative Gains):**

- **Top 10%** pelanggan berdasarkan skor → memuat **±27%** dari seluruh repeater (**lift ≈ 2.7×**).
- **Top 30%** → **±60%** repeater tertangkap.
- **Top 50%** → **±80%** repeater tertangkap.

Artinya tim dapat menangkap mayoritas repeater dengan menghubungi sebagian kecil pelanggan — jauh lebih efisien
daripada *follow-up* acak.

**Faktor pendorong repeat (feature importance):** umur, pekerjaan, karakteristik & harga produk (OTR, type series,
warna), lokasi (kecamatan), dan profil pembiayaan (rasio cicilan/DP, dealer).

*(Angka pasti dapat dilihat pada output sel notebook; nilai di atas berasal dari eksekusi dengan `SEED=42`.)*

## 5. Cara Menjalankan

### Opsi A — Google Colab (paling mudah)
1. Unggah `TrioTunggal_Python_SPARC.ipynb` ke [Google Colab](https://colab.research.google.com/).
2. `Runtime → Run all`. Notebook mengunduh data dari tautan resmi secara otomatis.

### Opsi B — Lokal
```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm jupyter
jupyter notebook TrioTunggal_Python_SPARC.ipynb
```
Lalu jalankan semua sel (`Run All`). Waktu eksekusi ± beberapa menit (termasuk unduh data & tuning).

## 6. Struktur Repositori

```
.
├── TrioTunggal_Python_SPARC.ipynb   # Notebook utama (EDA → preprocessing → modeling → output bisnis)
├── README.md                        # Dokumen ini
├── .gitignore                       # Mengecualikan dataset & output turunan (aturan kompetisi)
└── (SPARC_dataset.csv)              # TIDAK di-commit — diunduh saat notebook dijalankan
```

## 7. Reprodusibilitas

Seluruh proses acak (pembagian data, inisialisasi model, training, *tuning*) menggunakan **`SEED = 42`**,
sehingga hasil konsisten setiap kali dijalankan. Seluruh output pada notebook dihasilkan dari eksekusi kode.

## 8. Teknologi

`Python` · `pandas` · `NumPy` · `scikit-learn` · `LightGBM` · `matplotlib` · `seaborn`

---

**Tim TrioTunggal** — SPARC 2026.
