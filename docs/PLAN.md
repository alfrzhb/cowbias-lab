# MASTER IMPLEMENTATION PLAN

## CowBias Lab — Spurious Correlation Detection & Mitigation

**Versi:** 1.0 · 16 Oktober 2026  
**Status:** Rencana eksekusi; implementasi belum dimulai  
**Dokumen acuan:** `PRD_CowBias_Lab.md` v1.0 dan `Tugas_3.docx`  
**Output utama:** Satu repository Python yang runnable, eksperimen baseline vs treatment yang dapat direproduksi, dashboard visualisasi, dan laporan akademik.  
**Format eksekusi:** **Satu master plan, lima tahap terintegrasi**, tanpa memecah menjadi banyak dokumen per sprint.

---

## 1. Ringkasan Keputusan dan Hasil Akhir

**Tujuan:** Membangun mini-project akademik yang *menunjukkan secara terukur* bagaimana CNN dapat mempelajari shortcut berupa latar belakang (rumput/pantai) ketika mengenali sapi, lalu mengevaluasi apakah **background randomization** mengurangi ketergantungan tersebut.

**Dua model, satu perbandingan adil:**

- **Baseline:** dilatih pada 1.000 komposit sintetis berlabel sapi/bukan sapi, dengan korelasi kuat label–latar (97% pasangan dominan).
- **Treatment:** CNN **dengan arsitektur dan hyperparameter yang sama**, dilatih pada 1.000 foreground yang sama, namun latarnya diseimbangkan 50:50 pada setiap kelas.
- **Held-out evaluation:** validation seimbang 240 citra dan test seimbang 480 citra pada empat kelompok (`cow/grass`, `cow/beach`, `non_cow/grass`, `non_cow/beach`). Set test disusun sebagai pasangan counterfactual: objek yang sama pada dua background berbeda.
- **Bukti:** akurasi tiap kelompok, **worst-group accuracy (WGA)**, overall accuracy, confusion matrix, background flip rate, paired correctness, dan Grad-CAM. Minimal tiga seed: `42`, `123`, `2026`.
- **UI:** lima tab Streamlit, hanya membaca dan menghitung dari artefak yang dihasilkan pipeline nyata. Tidak ada angka eksperimen yang di-hardcode.

**Batas ilmiah:** Dataset MVP adalah **sintetis**. Hasilnya membuktikan perilaku pada lingkungan terkontrol, bukan kemampuan klasifikasi sapi di foto dunia nyata. Eksperimen foto nyata/Waterbirds adalah pekerjaan lanjutan, **bukan kriteria selesai MVP**.

### Definition of the shipped product

| Komponen | Berkas atau keluaran yang wajib ada | Bukti keberhasilan |
|---|---|---|
| Source Python | `src/cowbias/`, `scripts/`, `pyproject.toml` | Instalasi, CLI, dan tes lolos |
| Dataset & audit | Generator, manifest, crosstab per split | Empat grup, distribusi sesuai konfigurasi, tanpa source leakage |
| Trained CNN | Checkpoint baseline dan treatment per seed | Hasil inference berasal dari checkpoint |
| Evaluation | `predictions.csv`, `metrics.json`, ringkasan antarseed | WGA dan metrik lain dapat dihitung ulang |
| Explainability | Grad-CAM dua model, grafik perbandingan | Dari model terlatih dan sampel identik |
| Aplikasi | `app/streamlit_app.py` | Demonstrasi lokal, empty state, ekspor |
| Handoff akademik | `README.md`, metodologi, `docs/task-3-report.pdf` | Menjawab desain sistem, treatment, visualisasi, dan batasan |

---

## 2. Prinsip Eksekusi dan Batas Scope

1. **ML-first, UI-second:** Jangan membangun UI dengan data fiktif sebelum generator, model, dan evaluator menghasilkan artefak sah.
2. **Fair comparison:** Baseline dan treatment memiliki model, ukuran train, foreground IDs, optimizer, loss, batch size, epoch, resolusi, dan protokol evaluasi yang sama. Perbedaan terencana hanya pemilihan background saat training.
3. **Split by source before compositing:** Semua varian foreground yang sama dan semua varian dari aset background sumber yang sama harus berada pada split yang sama. Pair counterfactual pada test diperbolehkan **di dalam split test**, tidak lintas split.
4. **Held-out test is sacred:** Validasi dipakai untuk keputusan model/epoch, sedangkan test baru dipakai setelah desain ditetapkan. Jangan mengoptimalkan treatment berdasarkan test.
5. **Reproducible and auditable:** Setiap run punya konfigurasi, seed, checksum artefak/checkpoint, versi kode bila tersedia, metadata sumber, dan daftar prediksi baris-per-baris.
6. **Honest reporting:** Jika metrik treatment tidak naik, jangan mengubah angka. Jelaskan kemungkinan penyebab dan tampilkan failure case.
7. **CPU-first:** Training ringan dapat berjalan di CPU. GPU boleh mempercepat, tetapi bukan dependency untuk demo.
8. **No scope creep:** Tidak ada login, database, REST API, LLM/chatbot, object detection video, atau keharusan deployment publik untuk MVP.
9. **Dokumen PRD adalah source of truth:** Bila detail rencana tidak disebut PRD, tuliskan sebagai keputusan implementasi yang dapat direvisi, bukan sebagai tuntutan dosen.

### Hal yang secara eksplisit diminta pada soal

Soal meminta sistem yang menangani *fake correlation* serta **visualisasi model dan penjelasan treatment**. Karena itu, laporan dan dashboard harus memperlihatkan hubungan sebab-akibat desain eksperimen secara jelas: mengapa baseline berpotensi salah, apa intervensi pada data latih, serta apa dampaknya pada kelompok pengujian. Soal tidak mewajibkan GitHub atau Streamlit; keduanya digunakan sebagai sarana presentasi dan reproducibility.

---

## 3. Target Arsitektur dan Struktur Repository

### 3.1 Pipeline yang harus terwujud

```text
experiment.yaml + fixed seeds
            |
   generate source assets (foreground, backgrounds)
            |
    assign source IDs to train/val/test
            |
      compose and audit image datasets
            |
   +--------+-------------------+
   |                            |
baseline train set       balanced treatment set
97/3 label–background    50/50 per object label
   |                            |
   CNN baseline                CNN treatment
   |                            |
   +------------+---------------+
                |
      SAME held-out test with 4 groups
                |
      predictions per model, per seed
                |
       metrics / WGA / paired flip
                |
     confusion matrix / Grad-CAM plots
                |
     Streamlit reads real run artifacts
                |
       reviewer demo + final PDF
```

### 3.2 File layout target

```text
cowbias-lab/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── config/
│   └── experiment.yaml
├── app/
│   └── streamlit_app.py
├── src/cowbias/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── generate.py
│   │   ├── splits.py
│   │   ├── dataset.py
│   │   └── audit.py
│   ├── models/
│   │   └── cnn.py
│   ├── training/
│   │   ├── train.py
│   │   └── treatment.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── counterfactual.py
│   ├── explainability/
│   │   └── gradcam.py
│   └── visualization/
│       └── plots.py
├── scripts/
│   ├── prepare_data.py
│   ├── run_experiment.py
│   └── export_report.py
├── tests/
│   ├── test_dataset.py
│   ├── test_no_leakage.py
│   ├── test_metrics.py
│   ├── test_counterfactual.py
│   └── test_artifacts.py
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   ├── methodology.md
│   └── task-3-report.pdf
├── notebooks/
│   └── demo_colab.ipynb                 # optional
└── artifacts/                          # generated; ignored selectively
    └── runs/<run_id>/...
```

**Keputusan format:** Satu `docs/PLAN.md` ini merupakan rencana implementasi tunggal. Script CLI modular diperbolehkan karena itu **kode**, bukan pecahan dokumen plan.

### 3.3 Satu konfigurasi pusat

Mulai dari spesifikasi PRD berikut; semua angka dapat diubah melalui YAML, bukan dengan mengedit beberapa file source:

```yaml
project: cowbias-lab
dataset:
  image_size: 48
  model_image_size: 32
  train_size: 1000
  val_size: 240
  test_size: 480
  baseline_majority_ratio: 0.97
  treatment_background_ratio: 0.50
  negative_class: horse
training:
  seeds: [42, 123, 2026]
  epochs: 7
  batch_size: 64
  optimizer: adamw
  learning_rate: 0.001
  device: auto
artifacts:
  root: artifacts/runs
```

Angka konfigurasi ini adalah **default rencana**, bukan metrik performa. Jika scheduler atau perangkat membuat epoch awal tidak memadai, lakukan perubahan yang terdokumentasi berdasarkan **validation**, bukan test.

---

## 4. Lima Tahap Implementasi — Satu Alur Berurutan

### TAHAP 1 — Fondasi Repo dan Dataset yang Dapat Diaudit

**Tujuan:** Ada repo installable, dataset sintetis yang sah, dan bukti pencegahan kebocoran data sebelum model dilatih.

**Implementasikan dalam satu rangkaian:**

1. Inisialisasi repository `cowbias-lab`; tambahkan `pyproject.toml` (PyTorch, torchvision bila diperlukan, numpy, pandas, Pillow, matplotlib, Streamlit, PyYAML, pytest, Ruff). Tetapkan versi minimum yang didukung di file dependency dan dokumentasikan Python yang dipakai. Jangan commit environment lokal/secret.
2. Siapkan `config/experiment.yaml`, utilitas loading/validasi config, seed untuk `random`, NumPy, dan PyTorch, serta logging terstruktur. Semua scripts menerima `--config`; invalid YAML atau field hilang harus gagal dengan pesan jelas.
3. Bangun generator *foreground* dua kelas: `cow` dan `non_cow` (kuda pada MVP). Gunakan beberapa variasi bentuk, warna, proporsi, posisi, skala, dan orientasi. Bentuk foreground **bukan label teks** atau icon kelas yang trivially leaked. Sertakan mask foreground untuk compositing dan nanti dipakai sebagai referensi inspeksi Grad-CAM.
4. Buat generator latar `grass` dan `beach` dengan variasi tekstur, warna, horizon, pencahayaan, pola, dan parameter acak. **Jangan** menyisipkan metadata/label ke piksel citra. Aset latar harus berbeda sumber antara train, val, test.
5. **Bagi identitas aset sumber lebih dulu**, baru buat komposit. Setiap `foreground_id` dan `background_asset_id` memiliki satu `split` eksklusif. Variasi dari sumber yang sama tetap di split yang sama.
6. Bentuk training baseline: 500 sapi dan 500 bukan sapi. Sapi mayoritas `grass`, bukan sapi mayoritas `beach`, rasio total yang ditargetkan 97:3 dengan minoritas nyata tetap ada. Bentuk training treatment dengan **foreground training ID dan jumlah sampel identik**; untuk setiap label, `grass` dan `beach` masing-masing 50%.
7. Bentuk validation (60 citra/grup) dan test (120 citra/grup), grup seimbang. Untuk test, 480 citra berarti 240 foreground unik: 120 sapi + 120 bukan sapi; tiap foreground direalisasikan sekali pada rumput dan sekali pada pantai, dengan `pair_id` konsisten.
8. Ekspor file citra, mask bila digunakan, `manifest.csv`, dan `audit.json`. Audit harus memuat tabulasi empat grup per dataset/split, jumlah ID unik, pemeriksaan overlap, label balance, pair integrity, dan ringkasan parameter generator.

**Kontrak minimum manifest:**

| Field | Definisi |
|---|---|
| `image_id` | ID unik komposit |
| `foreground_id` | ID objek sumber, dipakai mencegah leakage |
| `background_asset_id` | ID latar sumber, dipakai mencegah leakage |
| `object_label` | `cow` atau `non_cow` |
| `background_label` | `grass` atau `beach` |
| `split` | `train`, `val`, atau `test` |
| `variant` | `baseline`, `treatment`, atau `shared_eval` |
| `pair_id` | ID pasangan pada val/test; boleh kosong pada train |
| `seed` | Seed generator |
| `file_path` | Relative path gambar yang diverifikasi |

**Tes otomatis sebelum lanjut:**

- Jumlah data dan distribusi label/latar sesuai spesifikasi; tidak ada grup uji kosong.
- Foreground/background **source ID disjoint** antara split, termasuk turunan dari aset sama.
- Setiap pair di test berisi tepat dua citra dengan label/foreground identik dan latar yang berbeda; mask, posisi, pose, dan geometry foreground identik dalam pair.
- Distribusi background treatment seimbang untuk **masing-masing** label, bukan hanya secara global.
- Baseline dan treatment memakai himpunan foreground train yang sama, jumlah records sama, dan karakteristik lain sebanding.
- Generator deterministik untuk seed tetap; berbeda seed memberi variasi terukur.
- Audit sampel render secara visual: tidak ada watermark, class text, garis komposit eksklusif pada satu label, atau background yang menghilangkan identitas objek.

**Exit gate tahap 1:** `prepare_data.py` selesai, crosstab dan manifest valid, `pytest` untuk generator/split lolos. **Jangan melatih model jika gate ini gagal.**

### TAHAP 2 — Melatih Baseline dan Treatment Secara Adil

**Tujuan:** Model CNN yang benar-benar belajar dan dua checkpoint sah untuk setiap seed, bukan prediksi yang diatur secara manual.

**Langkah implementasi:**

1. Implementasikan satu fungsi arsitektur CNN ringan (2–3 blok conv + pooling, classifier biner), input RGB 32×32. Baseline dan treatment **harus menggunakan kelas Python yang sama**.
2. Buat `Dataset`/`DataLoader` dari manifest; apply resize dan normalisasi konsisten. Pastikan **tidak ada flip/augmentasi tambahan** yang secara tidak sengaja membedakan baseline dari treatment selain penggantian background yang memang diuji.
3. Implementasikan `train.py`: loss klasifikasi biner, AdamW, loop train + validation, log epoch, akurasi/val WGA, pencatatan initial state/seed, penyimpanan best checkpoint. Pilih checkpoint dengan kriteria validation yang ditentukan sebelum mengevaluasi test; jika epoch fixed, tetap simpan kebijakan selection.
4. Implementasikan `treatment.py` sebagai logika **training dataset**, bukan filter gambar di dashboard. Pemilihan background tidak bergantung pada class label. Hasil train treatment terverifikasi 50:50 per label.
5. Buat `scripts/run_experiment.py` menjalankan kedua mode untuk seeds `42`, `123`, `2026`. Pastikan setiap pasangan model per seed menggunakan **identical initialization** (buat initial state sekali lalu load ke dua model) dan kondisi training setara.
6. Simpan checkpoint serta JSON konfigurasi, log history, metadata dataset/commit. Simpan checksum SHA-256 checkpoint, daftar foreground yang digunakan, arsitektur, seed, dan versi dependency.
7. Tambahkan smoke mode (subset kecil, satu epoch, satu seed) untuk memastikan seluruh lifecycle CPU bekerja tanpa perlu menjalankan eksperimen penuh setiap saat.

**Acceptance tahap 2:**

- Untuk masing-masing dari tiga seed terdapat checkpoint `baseline.pt` dan `treatment.pt` yang dapat di-`load_state_dict` lalu inferensi ulang.
- Kode training tidak mengakses label background sebagai fitur model, kecuali pemakaiannya untuk membangun distribusi input saat treatment.
- Hasil training log mencatat loss/metric, bukan sekadar keberhasilan proses.
- Konfigurasi baseline vs treatment berbeda **hanya** pada proses pembentukan latar training; variasi lain yang tidak dapat dihindari didokumentasikan.
- Tidak ada test-set feedback untuk mengubah training.

**Exit gate tahap 2:** Per seed tersedia 2 checkpoint + history + metadata; smoke test dan full run tidak menghasilkan NaN/shape error. Tidak wajib menjanjikan baseline/treatment dengan persentase akurasi tertentu.

### TAHAP 3 — Evaluasi Ilmiah, Counterfactual, dan Explainability

**Tujuan:** Menghasilkan bukti kuantitatif dan visual yang langsung menjawab bagian *treatment* dalam soal.

**Langkah implementasi:**

1. Buat evaluator yang mengambil checkpoint **dan** `manifest.csv` held-out yang sama. Urutan evaluation deterministik; kedua model melihat item test dengan `image_id` yang sama.
2. Hitung `overall_accuracy`, empat `group_accuracy`, `worst_group_accuracy = min(group_accuracy)`, confusion matrix, dan probabilitas kelas untuk setiap sampel. Tambahkan jumlah `n` per grup agar WGA bisa diaudit.
3. Hitung `background_flip_rate`: per `pair_id`, apakah prediksi kelas berubah ketika background diganti. Hitung juga **paired_correct_rate** agar model yang selalu salah/konsisten tidak dianggap bagus hanya karena flip rate rendah.
4. Tulis `predictions.csv` satu baris per `model x seed x image_id`. Tulis metrik per model/seed ke JSON; lakukan crosscheck agregasi dari CSV dengan isi JSON menggunakan assert / pytest.
5. Agregasi tiga seed: mean dan simpangan baku sampel untuk overall accuracy, WGA, group accuracy, flip rate, paired correctness. Simpan `summary.csv`/`summary.json`; cantumkan jumlah seed dan toleransi nondeterminism bila relevan.
6. Generate grafik accuracy-by-group yang memperlihatkan baseline/treatment berdampingan, confusion matrices, contoh prediksi pasangan cow/grass ↔ cow/beach serta non_cow/grass ↔ non_cow/beach.
7. Implementasikan Grad-CAM pada feature map conv terakhir. Tampilkan pasangan gambar **yang sama** untuk kedua checkpoint; nyatakan kelas target secara eksplisit (disarankan `true class` agar heatmap dibandingkan pada target sama). Buat overlay tanpa mengubah input model.
8. Tulis `docs/methodology.md`: tujuan, kontrol eksperimen, apa yang diubah oleh treatment, rumus dan interpretasi metrik, sumber validitas internal/eksternal, penjelasan bahwa Grad-CAM bukan bukti kausal tunggal.

**Contoh kontrak artefak:**

```text
artifacts/runs/<run_id>/
├── run_config.json
├── data/manifest.csv
├── data/audit.json
├── seed_42/
│   ├── baseline.pt
│   ├── treatment.pt
│   ├── baseline_history.csv
│   ├── treatment_history.csv
│   ├── predictions.csv
│   ├── metrics.json
│   └── figures/
│       ├── accuracy_by_group.png
│       ├── confusion_matrix.png
│       ├── counterfactual_examples.png
│       └── gradcam_comparison.png
├── seed_123/...
├── seed_2026/...
├── summary.csv
└── summary.json
```

```text
predictions.csv required columns:
run_id,seed,model,image_id,pair_id,y_true,background_label,
predicted,prob_cow,correct
```

**Exit gate tahap 3:** Semua metrik dapat dihitung ulang dari `predictions.csv`, grafik dan heatmap benar-benar berasal dari checkpoint, ringkasan tiga seed tersedia, dan temuan ditulis apa adanya (positif, campuran, atau negatif).

### TAHAP 4 — Dashboard Streamlit sebagai Media Demonstrasi

**Tujuan:** Reviewer dapat memahami keseluruhan eksperimen dalam demo sekitar 3–5 menit tanpa harus membaca source code.

**Lima tab sesuai PRD:**

| Tab | Detail yang wajib tersedia | Interaksi & error handling |
|---|---|---|
| **Overview** | Masalah, hipotesis, arsitektur, penjelasan baseline vs treatment | Banner **SYNTHETIC EXPERIMENT** selalu jelas |
| **Dataset Explorer** | Empat kelompok, jumlah per split, sampel dataset dan metadata | Filter split/label/latar; tidak menyajikan pasangan palsu |
| **Model Comparison** | Dua prediksi checkpoint berdampingan pada input yang sama | Pilih seed dan `pair_id`, toggle grass/beach; benar/salah + probabilitas |
| **Evaluation** | WGA, overall, accuracy by group, confusion matrix, mean ± SD | Pilih run/seed; setiap angka berasal dari CSV/JSON valid |
| **Explainability & Export** | Grad-CAM, treatment explanation, keterbatasan, download | Export CSV/JSON/PNG; target class heatmap jelas |

**Aturan implementasi UI:**

1. Dashboard membaca `artifacts/runs/<run_id>`; sertakan dropdown run dan seed, validasi keberadaan file, tipe kolom, dan cek bahwa model/checkpoint cocok dengan metrics yang dipilih.
2. Jika folder hasil belum ada, tampilkan **“Experiment not yet run”**, cara menjalankan CLI, tanpa angka placeholder.
3. Gunakan cache Streamlit hanya untuk loading checkpoint/dataset, bukan hasil fiktif. Prediksi interactive harus benar-benar menjalankan inference checkpoint atau membaca `predictions.csv` tervalidasi untuk sampel test yang tersedia.
4. Perbandingan background memakai pair asli: perubahan latar **tidak** boleh mengganti identitas foreground. Tampilkan ID, ground truth, background, dan run seed.
5. UI didesain bersih, responsif, sederhana, tanpa kebutuhan CSS kompleks. Fokus pada bukti ilmiah.
6. **Tidak ada tombol Training di UI untuk MVP.** Training dijalankan offline; dashboard hanya konsumsi artefak.
7. Halaman export juga memuat disclaimer: hasil dataset sintetis, bukan bukti performa pada foto nyata; WGA dan flip rate harus ditafsirkan bersama.

**Exit gate tahap 4:** aplikasi start lokal, navigasi lima tab tanpa exception, empty state valid, dua model tampil benar pada sampel yang sama, dan ekspor file bisa dibaca ulang.

### TAHAP 5 — Pengujian Menyeluruh, Laporan Akademik, dan Handoff GitHub

**Tujuan:** Repository siap diaudit, direproduksi, dipresentasikan, dan—setelah ditinjau—di-push.

**Urutan finishing:**

1. Jalankan `pytest -q`, Ruff, generator/audit, training dan evaluation smoke, serta smoke dashboard; catat semua perintah dan status. Untuk Stage 5, gunakan bukti full run tiga seed asli Stage 2-3 yang sudah diverifikasi; jangan retrain enam model atau mengulang full held-out evaluation hanya untuk QA.
2. Audit sumber data dan angka: cocokan summary CSV/JSON, screenshot dashboard, tabel laporan, dan prediksi. Semua harus menunjuk run yang sama dan identitas seed yang jelas.
3. Periksa edge cases: folder artefak hilang, CSV tidak lengkap, checkpoint mismatch, model menghasilkan satu kelas terus-menerus, grup uji kosong, gambar rusak, duplicate IDs, dan anomali leakage. Pesan kegagalan harus informatif.
4. Susun README: overview masalah; diagram; install; quick start; contoh CLI; screenshot hasil aktual; interpretasi treatment dan WGA; batasan synthetic; licence/attribution; referensi ilmiah.
5. Susun PDF **jawaban tugas final** di `docs/task-3-report.pdf` **setelah** run selesai. Gunakan hasil yang diambil otomatis dari `summary.json` dan grafik run, bukan angka dari PDF prototipe sebelumnya. Outline: pendahuluan masalah → desain sistem → dataset dan baseline → treatment → prosedur eksperimen → hasil per kelompok dan counterfactual → Grad-CAM → diskusi & limitasi → kesimpulan → referensi.
6. Tambahkan `docs/PRD.md` dan `docs/PLAN.md` ke repo. Pastikan jumlah file dokumentasi tidak berlebihan; metodologi dan laporan punya fungsi berbeda.
7. `.gitignore` mengecualikan virtualenv/cache, dataset generated yang besar, checkpoints besar, credential, dan file sementara. Artefak kecil yang memang ingin dijadikan demonstrasi dapat ditrack jika ukurannya wajar; bila checkpoint besar, berikan metode reproduksi/unduhan yang sah.
8. Jalankan fresh-clone check pada environment bersih atau direktori baru: install, sample generation, smoke run, dashboard. Tidak boleh membutuhkan file lokal tersembunyi milik developer.
9. Setelah seluruh gate lulus, commit ringkas dan jelas. Push GitHub dilakukan hanya bila memang diminta pada tahap eksekusi dan akses tersedia; membuat plan **tidak** otomatis membuat repo/push.

**Exit gate tahap 5:** README mengikuti kondisi aktual, PDF baru berisi hasil nyata run repo, fresh-clone demo berjalan, source dan artefak yang didistribusikan dapat ditelusuri. Implementasi Stage 5 diserahkan melalui PR untuk review independen; tidak auto-merge. Preservasi arsip durabel ditandai PENDING bila belum tersedia tujuan dengan kontrol akses yang diotorisasi. Lihat `docs/reproducibility.md` dan manifest bukti.

---

## 5. Command Contract untuk AI Coding Agent

Perintah berikut merupakan **interface yang wajib diimplementasikan**, bukan klaim perintah sudah dapat dieksekusi saat dokumen dibuat.

```bash
# setup
python -m venv .venv
# aktifkan .venv sesuai OS
python -m pip install -e '.[dev]'

# generate + audit data
python scripts/prepare_data.py --config config/experiment.yaml

# smoke run (developer experience; harus diimplementasikan)
python scripts/run_experiment.py --config config/experiment.yaml --smoke

# full run: 3 seeds, 2 model per seed, metrics + plot
python scripts/run_experiment.py --config config/experiment.yaml --seeds 42 123 2026

# tests
pytest -q
ruff check .

# dashboard
streamlit run app/streamlit_app.py

# PDF report dari hasil run; pilih run_id aktual
python scripts/export_report.py --run-id <run_id>
```

**CLI behavior:** `--help` harus tersedia, invalid argument gagal non-zero, path artifact ditampilkan, dan `--run-id` menunjuk run spesifik. Jika rerun folder bertabrakan, buat ID baru atau minta flag overwrite eksplisit; jangan diam-diam menimpa bukti.

## 6. Verification Matrix dan Definition of Done Final

| ID | Area | Uji / kondisi lulus | Jenis |
|---|---|---|---|
| QG-01 | Dataset | Train, val, test ada dan jumlah total sesuai config | Automated |
| QG-02 | Leakage | Source foreground dan background disjoint antar split | Automated |
| QG-03 | Bias control | Baseline 97:3 dan treatment 50:50 **per kelas** sesuai config | Automated |
| QG-04 | Fairness | Arsitektur/hyperparameter/jumlah train konsisten | Automated + audit |
| QG-05 | Checkpoints | Baseline dan treatment loadable untuk tiga seed | Automated |
| QG-06 | Metrics | Group accuracy dan WGA dihitung dari predictions, konsisten | Automated |
| QG-07 | Counterfactual | Setiap pair objek identik dan kedua background lengkap | Automated + visual |
| QG-08 | Explainability | Grad-CAM dari layer conv dan checkpoint tepat | Automated + visual |
| QG-09 | Dashboard | Lima tab, empty state, artifact validation dan ekspor | Manual smoke |
| QG-10 | Reporting | Tabel/angka PDF sama dengan output run; synthetic disclaimer | Manual crosscheck |
| QG-11 | DX/reproducibility | Clean setup, CLI, seed/config, README, run manifest | Manual + automated |
| QG-12 | Integrity | Tidak ada metrik hardcoded atau klaim foto nyata | Code review |

**Definisi DONE:** Semua QG-01 sampai QG-12 lulus. **Bukan** syarat bahwa WGA treatment pasti melebihi baseline; kalau hasilnya tidak sesuai hipotesis, tetap gunakan data aktual dan jelaskan apa yang teramati. Perbaikan WGA adalah *research hypothesis*, bukan angka yang boleh direkayasa demi kelulusan fitur.

### Negative / edge-case tests yang wajib

- Pembuatan dataset dengan rasio tidak valid (`-0.1` atau `1.5`) gagal cepat.
- Dua sampel dengan `foreground_id` sama pada train dan test memicu audit error.
- Asset background yang sama muncul di dua split memicu audit error.
- Satu `pair_id` hanya memiliki satu background memicu pair validation error.
- Group count nol menghasilkan pesan eksplisit, bukan pembagian nol.
- Baris `predictions.csv` yang duplikat/tidak lengkap ditolak.
- File model tidak ada atau mismatched dengan run menampilkan empty/error state, tidak menampilkan hasil dari run lain.
- Counterfactual flip rate 0 karena model selalu menebak satu kelas tidak dianggap sukses tanpa paired accuracy.

---

## 7. Risiko Teknis, Alternatif, dan Aturan Pengambilan Keputusan

| Risiko | Tanda awal | Respons yang harus dilakukan |
|---|---|---|
| Model tidak mempelajari bentuk objek | Train/val accuracy rendah untuk semua grup | Inspeksi sampel, bentuk/warna/resize, cek label; perbaiki generator tanpa menyentuh test selection |
| Baseline tidak menunjukkan shortcut | Perbedaan grup kecil, flipping rendah sejak awal | Audit rasio bias, shortcut visibility, kuat/lemahnya sinyal object; dokumentasikan kondisi; jangan manipulasi angka |
| Treatment masih bergantung pada background | Minority group accuracy tetap rendah | Audit independensi latar, border artifacts, jumlah contoh, dan hyperparameter berdasar validation |
| Leakage tak terlihat | Score terlalu tinggi/identik | Tambah tes overlap source-family, hash aset, revisi split sebelum komposit |
| CNN di 32×32 terlalu sederhana | Salah mengenali sapi/kuda secara sistematis | Pertahankan fokus sintetis, tingkatkan desain foreground/arsitektur secara simetris dan laporkan revisi |
| Variasi antarseed tinggi | Perbandingan tidak stabil | Laporan mean ± SD, distribusi tiap seed dan error bars |
| Grad-CAM dianggap bukti utama | Heatmap bagus namun metrics tidak membaik | Utamakan WGA, group metrics, dan paired counterfactual dalam kesimpulan |
| Beban komputasi tinggi | CPU full run lambat | Optimalkan data loading, image cache, smoke mode; hindari mengurangi metode evaluasi |
| Artefak tidak cocok | Grafik dan JSON beda | Validasi `run_id`, config hash, checksum, rebuild outputs dari satu run |

**Aturan eskalasi untuk agent:** Jika perubahan terencana menyentuh dataset size, label classes, model family, treatment type, evaluasi empat grup, atau cara pemilihan checkpoint, dokumentasikan perubahan terlebih dahulu dalam bagian *decision log* pada `docs/PLAN.md`, lalu lakukan perubahan yang sama secara adil untuk kedua model. Jangan diam-diam mengubah scope untuk mendapat skor tinggi.

---

## 8. Urutan Kerja, Checkpoint, dan Handoff

Gunakan satu cabang kerja. Eksekusi lima tahap **secara berurutan** dengan checkpoint kode yang mudah ditinjau; jumlah commit fleksibel, bukan lima repository atau lima PRD.

| Tahap | Dependency | Artefak checkpoint | Boleh lanjut jika |
|---|---|---|---|
| 1. Repo + dataset | PRD | generator, manifest, audit, test split | Tidak ada leakage; empat grup benar |
| 2. Models + training | Tahap 1 | 6 checkpoints, logs, config | Kedua model per seed berhasil diinferensi |
| 3. Evaluation + XAI | Tahap 2 | CSV, JSON, figures, metodologi | Metrik dihitung ulang dan plot valid |
| 4. Dashboard | Tahap 3 | 5-tab local app | Semua angka/visual dari artifact asli |
| 5. QA + report | Semua | README, PDF, repo siap handoff | 12 quality gates lulus |

**Prioritas dalam keadaan waktu terbatas:** jangan hilangkan validitas eksperimen. Kurangi polishing UI dan tambahan grafik lebih dulu; pertahankan dataset audit, dua model sungguhan, pengujian empat grup, treatment nyata, dan satu visualisasi model yang dijelaskan. Untuk spesifikasi penuh PRD, kelima tahap harus selesai.

### Handoff untuk AI coding agent

> Implementasikan repository `cowbias-lab` end-to-end berdasarkan `docs/PRD.md` dan `docs/PLAN.md`. Gunakan **satu master plan** ini; selesaikan Tahap 1 hingga Tahap 5 secara berurutan. Wajib menjaga kontrol eksperimen baseline vs treatment, mencegah leakage sejak source IDs, menghasilkan model CNN terlatih dan artefak evaluasi nyata (3 seeds), serta membangun dashboard Streamlit yang membaca hasil aktual. Jangan menggunakan angka/plot hardcoded dan jangan mengklaim performa foto nyata dari data sintetis. Jalankan quality gates di tiap tahap, dokumentasikan asumsi/revisi, dan pada akhir pekerjaan keluarkan ringkasan file yang dibuat, perintah dijalankan, hasil tes, metrik aktual, batasan, serta status commit/push.

---

## 9. Catatan Keterlacakan dan Status

- **Sumber kebutuhan akademik:** `Tugas_3.docx` — sistem menerapkan treatment pada fake correlation; visualisasi model dengan penjelasan treatment.
- **Sumber spesifikasi produk:** `PRD_CowBias_Lab.md` v1.0 — arsitektur CNN sintetis, rasio bias, empat kelompok, tiga seed, WGA, Grad-CAM, lima tab Streamlit, non-goals.
- **Keputusan implementasi tambahan di master plan:** detail format YAML, opsi smoke test, run ID, pola error handling, checksum, langkah QA, dan single-plan handoff. Semua adalah pedoman engineering untuk memenuhi PRD, bukan kewajiban eksplisit dari soal.
- **Status saat master plan dibuat:** dokumen spesifikasi dan rencana tersedia; **kode/repository baru belum otomatis dibuat, diuji, atau di-push** hanya karena rencana ini selesai.