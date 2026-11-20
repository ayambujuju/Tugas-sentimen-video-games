# Klasifikasi Sentimen Ulasan Video Games (Kasus B)

Model biner (negatif/positif) untuk ulasan Amazon Reviews 2023 kategori
Video_Games, disajikan lewat REST API (FastAPI).

## Data
- Sumber: Amazon Reviews 2023, Video_Games, hanya ulasan sampai 31 Des 2019.
- Label: rating 1-2 = negatif, 4-5 = positif, rating 3 dibuang.
- Input: judul + teks ulasan digabung.
- File dibaca per baris (streaming) dan diambil sampel acak ~8%,
  menghasilkan 210.508 ulasan setelah duplikat dibuang (positif 81,3%).

## Pembagian data
Split berdasarkan waktu: 80% ulasan terlama = train (168.406), 20% terbaru
= validasi (42.102, Agustus 2018 - 2019). Alasan: di dunia nyata model dipakai
untuk ulasan baru, dan split acak membuat skor terlalu optimis. Duplikat
dibuang sebelum split agar tidak bocor. Tuning memakai holdout 15% data
train terbaru, sehingga data validasi hanya dipakai sekali untuk skor akhir.

## Metode dan alasan
TF-IDF (unigram + bigram, 200 ribu fitur) + Logistic Regression dengan
class_weight="balanced". Dipilih karena ringan, cepat, mudah dijelaskan, dan
cocok untuk deploy di server gratis. class_weight menangani ketimpangan kelas
(81% positif) agar macro-F1 tidak hanya bagus di kelas mayoritas.

## Skor di data validasi
| Versi | Keterangan | Macro-F1 | F1 negatif | F1 positif |
|---|---|---|---|---|
| v1 | baseline, C=1 | 0,9262 | 0,88 | 0,97 |
| v2 | tuning, C=10 (dipakai API) | **0,9345** | 0,89 | 0,97 |

## MLOps
- Versioning: model_v1.pkl, model_v2.pkl; semua run tercatat di MLflow
  (jalankan `mlflow ui --backend-store-uri sqlite:///mlflow.db`).
- Observability: tiap prediksi dicatat di predictions.log
  (waktu, input, label, probabilitas, versi model).
- Re-training: retrain.py menghitung macro-F1 model aktif pada data baru.
  Jika di bawah 0,85, model dilatih ulang menjadi versi baru dan hanya
  dipakai jika lebih baik dari yang lama.

## Menjalankan
    pip install -r requirements.txt
    uvicorn app:app --reload
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
      -d '{"title":"Great","text":"Loved this game"}'
Endpoint: POST /predict, GET /health, GET /model-info.
Demo online: https://tugas-sentimen-video-games.onrender.com