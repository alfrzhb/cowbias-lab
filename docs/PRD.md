# PRODUCT REQUIREMENTS DOCUMENT (PRD)

## CowBias Lab — Spurious Correlation Detection & Mitigation

**Versi:** 1.0 (MVP specification)  
**Tanggal:** 16 Oktober 2026  
**Status:** Siap dijadikan acuan implementasi; belum berarti seluruh fitur sudah dibuat  
**Jenis produk:** Mini-project akademik, eksperimen Deep Learning, dan dashboard demonstrasi  
**Platform:** Python + PyTorch + Streamlit; repository GitHub; eksekusi lokal / Google Colab untuk pelatihan  
**Mata kuliah:** Kecerdasan Buatan, Magister Informatika, UIN Sunan Kalijaga  
**Dokumen sumber:** `Tugas_3.docx` (Tugas 3, 14 Oktober 2026)

---

## 1. Executive Summary

**CowBias Lab** adalah aplikasi eksperimen yang menunjukkan bagaimana model klasifikasi citra bisa mempelajari *spurious correlation* (korelasi semu) antara label objek dan latar belakang. Kasus utama mengikuti soal: model melihat sapi yang lazim berada di rerumputan, lalu berpotensi keliru ketika sapi dipindahkan ke pantai.

Produk membandingkan **dua CNN dengan arsitektur identik**:

- **Baseline:** dilatih pada data yang mengandung korelasi semu kuat (sapi cenderung berlatar rumput; objek bukan sapi cenderung berlatar pantai).
- **Treatment:** dilatih pada foreground yang setara, tetapi latar belakangnya diacak dan diseimbangkan antarlabel (*background randomization*).

Kedua model diuji pada empat kombinasi objek–latar, dilengkapi perbandingan prediksi *counterfactual*, confusion matrix, metrik **worst-group accuracy (WGA)**, dan visualisasi **Grad-CAM**. Pelatihan dan evaluasi menggunakan data sintetis pada MVP agar eksperimen terkendali dan dapat direproduksi; hasilnya **tidak diklaim** sebagai kinerja pada foto sapi asli.

**Prinsip produk:** Bukti perubahan perilaku model harus berasal dari eksperimen yang berjalan, bukan angka yang ditulis statis di dashboard.

## 2. Landasan Soal dan Interpretasi

### 2.1 Instruksi asli yang harus dipenuhi

Dari dokumen tugas:

> “Mendesain sistem yang mengimplementasikan perbaikan pada fake correlation ... sistem mengenali sapi ... latar belakang rerumputan ... gambar sapi di pantai ... sistem yang diimplementasikan bisa menangani hal tersebut.”

> “Students should hand in a visualization model with an explanation of the treatment on system.”

### 2.2 Pemetaan instruksi menjadi deliverable

| Frasa pada soal | Makna operasional | Bukti pada proyek |
|---|---|---|
| Design a system | Sistem AI memiliki alur data, model, dan keluaran jelas | Diagram arsitektur dan dokumentasi |
| Implements a treatment | Ada intervensi yang benar-benar dijalankan | Modul background randomization dan model treatment |
| Fake correlation | Kinerja diuji ketika hubungan label–latar berubah | Matriks evaluasi empat kelompok dan uji counterfactual |
| Visualization model | Perilaku/model divisualisasikan | Grafik per kelompok, Grad-CAM, dan prediksi |
| Explanation of treatment | Dijelaskan mengapa dan di mana treatment bekerja | Penjelasan dashboard, README, laporan PDF |

**Penegasan:** Soal tidak secara eksplisit mewajibkan web app, Streamlit, GitHub, deployment, atau LLM. Semua itu adalah keputusan implementasi tambahan. Sub-CPMK menyebut DL/LLM/Generative AI sebagai capaian pembelajaran, tetapi soal spesifik ini berfokus pada treatment korelasi semu. MVP menggunakan Deep Learning tanpa menambahkan LLM hanya untuk memperbanyak fitur.

## 3. Masalah, Hipotesis, dan Nilai Produk

**Problem statement.** Model dapat mencapai akurasi tinggi pada distribusi data dominan sambil memakai latar belakang sebagai petunjuk yang tidak stabil. Ketika latar berubah, prediksi dapat berubah meskipun objek tetap sama. Ini merupakan contoh *shortcut learning* dan generalisasi yang rapuh.

**Hipotesis eksperimen.** Dengan menyeimbangkan latar untuk setiap label pada data latih, model memiliki insentif lebih kecil untuk menjadikan latar sebagai shortcut, sehingga performa pada kelompok minoritas dan kestabilan prediksi antarlatar *berpotensi* meningkat. Hipotesis harus diuji, bukan diasumsikan benar.

**Nilai bagi pengguna.** Pengguna dapat melihat penyebab kegagalan, menjalankan metode treatment, memahami metrik yang relevan, dan mempresentasikan bukti visual yang dapat diaudit.

## 4. Sasaran, Indikator Keberhasilan, dan Non-goals

### 4.1 Sasaran produk

1. Menghadirkan demonstrasi end-to-end dua model (baseline dan treatment) yang benar-benar dilatih.
2. Memperlihatkan efek korelasi semu pada empat kelompok objek–latar.
3. Menyediakan evaluasi yang adil, reproduktif, dan mudah diperiksa dosen.
4. Menghasilkan visualisasi dan dokumen yang dapat disertakan sebagai jawaban tugas.
5. Menyimpan eksperimen dalam repo yang mudah dijalankan developer/AI coding agent.

### 4.2 KPI dan kriteria ilmiah

| Indikator | Target MVP | Catatan |
|---|---|---|
| Reproducibility | Run dengan seed dan konfigurasi tercatat; hasil dapat dihitung ulang | Wajib |
| Empat kelompok uji | Keempatnya ada, setiap kelompok memiliki jumlah uji yang sama | Wajib |
| Worst-group accuracy | Terhitung pada baseline dan treatment | Metrik utama, bukan janji nilainya |
| Counterfactual flip rate | Terhitung untuk pasangan foreground sama, background berbeda | Wajib |
| Treatment | Mengubah distribusi latar pada data *latih* dan terdokumentasi | Wajib |
| Eksperimen berulang | Minimal 3 seed untuk ringkasan mean dan simpangan baku | Wajib sebelum klaim akhir |
| Visualisasi | Perbandingan kelompok, confusion matrix, contoh Grad-CAM | Wajib |
| Target hasil ilmiah | Diupayakan WGA treatment > WGA baseline | Jika tidak tercapai, tampilkan kegagalan secara jujur |

**Penting:** Produk dianggap memenuhi spesifikasi jika pipeline bekerja dan hasil dievaluasi secara transparan, bahkan bila suatu percobaan treatment tidak memperbaiki metrik. Tidak boleh mengubah angka agar terlihat berhasil.

### 4.3 Di luar scope MVP

- Deteksi objek dengan bounding box pada video real-time.
- Klaim mampu mengenali seluruh foto sapi di dunia nyata.
- Upload gambar sembarang dengan janji akurasi tinggi.
- Segmentasi foto alami otomatis sebagai fitur wajib.
- Integrasi LLM/chatbot, user authentication, backend API, database, cloud GPU, atau CI/CD kompleks.
- Training langsung dari dashboard untuk pengunjung umum; pelatihan dilakukan melalui CLI/notebook.

## 5. Persona dan User Journey

| Persona | Kebutuhan utama |
|---|---|
| Mahasiswa/pengembang | Menjalankan eksperimen dan menyiapkan bukti tugas |
| Dosen/reviewer | Memahami treatment dan mengecek apakah perbandingan adil |
| Pengunjung GitHub | Mencoba demonstrasi dan memahami batasannya |

**Happy path:** `Setup repo → Generate dataset → Audit distribusi label/latar → Train baseline → Train treatment → Evaluate test set → Generate visualisasi → Buka dashboard → Export hasil → Susun PDF.`

**Alur demo reviewer (3–5 menit):** Buka halaman overview, lihat empat contoh kombinasi, pilih satu sapi dengan latar rumput lalu bandingkan prediksinya ketika dipindahkan ke pantai, lihat grafik WGA dan heatmap Grad-CAM, kemudian baca penjelasan treatment dan keterbatasan dataset sintetis.

## 6. Scope Produk dan Prioritas

| ID | Fitur | Prioritas | Output |
|---|---|---|---|
| FR-01 | Generator dataset sintetis + metadata | P0 | Image files / manifest CSV |
| FR-02 | Audit distribusi empat grup | P0 | Tabel frekuensi / validasi |
| FR-03 | Pipeline training baseline CNN | P0 | Checkpoint dan training log |
| FR-04 | Background randomization | P0 | Dataset treatment seimbang |
| FR-05 | Pipeline training treatment CNN | P0 | Checkpoint dan log sebanding |
| FR-06 | Evaluasi grup + WGA | P0 | metrics.json, predictions.csv |
| FR-07 | Counterfactual paired comparison | P0 | Perubahan prediksi per pasangan |
| FR-08 | Grad-CAM kedua model | P0 | Overlay prediksi pada citra yang sama |
| FR-09 | Dashboard Streamlit | P0 | Demo yang membaca artefak aktual |
| FR-10 | Export dan dokumentasi | P0 | Grafik, tabel, README, laporan |
| FR-11 | Batch foto sapi nyata dan segmentasi | P1 | Eksperimen eksternal terpisah |
| FR-12 | ResNet/MobileNet pretrained, benchmark Waterbirds | P2 | Perluasan eksperimen, bukan MVP |

### 6.1 Acceptance criteria untuk fitur kritis

**FR-01 — Dataset.** Sistem membangkitkan dua label foreground (`cow`, `non_cow`) dan dua jenis latar (`grass`, `beach`). Setiap record mempunyai `image_id`, `foreground_id`, `object_label`, `background_label`, `split`, `seed`, dan path. Foreground ID serta aset latar sumber tidak boleh melintasi train/validation/test.

**FR-02 — Audit.** Tabel crosstab per split tampil. Distribusi training baseline memiliki korelasi dominan yang dikonfigurasi; dataset treatment memiliki perbandingan latar 50:50 untuk masing-masing label; validation/test empat grup seimbang.

**FR-03/05 — Model.** CNN baseline dan treatment memakai kode arsitektur, optimizer, epoch, ukuran input, batch size, dan jumlah sampel yang sama. Perbedaan yang disengaja hanya skema pembangkitan/pemasangan latar pada data latih. Konfigurasi dan seed direkam.

**FR-04 — Treatment.** Latar diganti **sebelum pelatihan** (atau selama loading training) dengan nilai yang tidak bergantung pada label objek. Bukan sekadar menambah filter gambar di UI. Identitas foreground dipertahankan; tekstur dan artefak komposit tidak boleh diam-diam membocorkan label.

**FR-06/07 — Evaluasi.** Prediksi berasal dari checkpoint yang dimuat. Perhitungan WGA = nilai terendah dari akurasi keempat grup. Counterfactual membandingkan gambar dengan objek/pose sama dan latar berbeda. Uji tidak memakai data training.

**FR-08 — Grad-CAM.** Heatmap dihasilkan dari layer konvolusi terpilih untuk target kelas yang jelas, pada contoh yang sama untuk kedua model. Teks UI menyatakan Grad-CAM adalah visualisasi pendukung, **bukan** treatment dan **bukan** pembuktian kausal tunggal.

**FR-09/10 — Dashboard dan ekspor.** Dashboard tidak boleh menampilkan angka placeholder sebagai hasil; bila artefak belum tersedia, tampilkan status `Experiment not yet run`. Unduhan hasil mencakup CSV/JSON/grafik dan metadata percobaan.

## 7. Spesifikasi Dataset dan Protokol Eksperimen

### 7.1 Definisi label dan grup

- `Y = 1`: sapi (`cow`); `Y = 0`: bukan sapi (`non_cow`), pada MVP gunakan kategori pembanding yang konsisten (misalnya ilustrasi kuda).
- `B = 0`: rerumputan (`grass`); `B = 1`: pantai (`beach`).
- **Empat grup evaluasi:** `non_cow/grass`, `non_cow/beach`, `cow/grass`, `cow/beach`.

Penggunaan kuda sebagai kelas negatif adalah **keputusan desain proyek**, bukan ketentuan dari dosen. Atribut nonlatar seperti ukuran objek, ketajaman, posisi, warna, dan gaya rendering dibuat bervariasi dan tidak boleh menjadi penanda otomatis label.

### 7.2 Konfigurasi default MVP (target implementasi)

| Variabel | Konfigurasi awal |
|---|---|
| Resolusi citra | 48 × 48 RGB; model menerima 32 × 32 RGB |
| Sampel training / eksperimen | 1.000 |
| Sampel validation | 240 (60 per grup) |
| Sampel test | 480 (120 per grup; berbentuk pasangan counterfactual) |
| Class balance | 50:50 sapi : bukan sapi |
| Latar baseline training | 97% pasangan dominan; 3% pasangan minoritas |
| Latar treatment training | 50:50 untuk setiap kelas, bebas dari label |
| Seeds | 42, 123, 2026 (dapat dikonfigurasi) |
| Arsitektur | CNN ringan 2–3 blok konvolusi |
| Optimizer | AdamW; parameter, termasuk learning rate, tersimpan di config |
| Epoch awal | 7 (dapat dikonfigurasi; hasil dipilih tanpa melihat test) |

Angka di atas adalah spesifikasi rencana; bukan hasil eksekusi repo yang belum dibuat. Konfigurasi sengaja dekat dengan simulasi pada PDF jawaban sebelumnya agar konsisten, tetapi angka hasil lama tidak otomatis dianggap terulang.

### 7.3 Pembagian data dan pencegahan leakage

1. Buat himpunan foreground dan himpunan background asset terpisah untuk train, val, test **sebelum compositing**.
2. Semua turunan dari foreground yang sama hanya berada di satu split; berlaku juga untuk tekstur/aset latar sumber.
3. Model baseline dan treatment memakai kumpulan foreground training yang identik dan jumlah komposit yang setara.
4. Validation dan test dibuat seimbang pada keempat grup, tanpa mengganti distribusinya agar memperbaiki skor.
5. Pair test harus menggunakan foreground identik dengan dua latar; ID pasangan disimpan.
6. Jangan memilih arsitektur, epoch, atau hyperparameter berdasarkan performa test. Test dipakai untuk evaluasi final.

### 7.4 Mekanisme treatment

Untuk mask foreground `M`, gambar objek `F`, dan latar `B`:

`X_composite = M * F + (1 - M) * B`

Pada baseline, probabilitas latar dibuat sangat berkorelasi dengan kelas. Pada treatment, pemilihan latar dibuat independen dari kelas: `P(background | cow) ≈ P(background | non_cow) = 0.5`. Identitas objek dan label kebenaran tidak berubah ketika latar diganti.

### 7.5 Fair comparison

- Backbone, fungsi loss, optimizer, batch, epoch/early stopping, jumlah training records, dan prosedur evaluasi sama.
- Setiap seed menggunakan kondisi awal model yang sebanding; seed dan identitas artefak dicatat.
- Bandingkan metrik per seed, lalu agregasikan mean ± standard deviation.
- Jangan menggunakan test set untuk memilih model terbaik.

## 8. Evaluasi, Formula, dan Bukti

**Primary:** worst-group accuracy — `WGA = min_g(correct_g / total_g)` dengan `g` salah satu dari empat kombinasi kelas/latar.

**Secondary:** overall accuracy, accuracy setiap grup, confusion matrix, background flip rate (fraksi pasangan foreground yang berubah prediksi ketika latarnya berubah), serta probabilitas prediksi untuk pasangan tertentu.

**Analisis counterfactual:** pasangan identik selain background digunakan sebagai uji sensitivitas latar. Flip rate lebih rendah **tidak otomatis** lebih baik jika prediksi konsisten tetapi sama-sama salah; laporkan bersama akurasi pasangan.

**Grad-CAM:** overlay dari target kelas yang sama pada baseline dan treatment, untuk gambar yang sama; interpretasi hanya berupa indikasi area penting pada prediksi.

**Syarat laporan:** tabel hasil baseline/treatment, nilai WGA per seed, contoh pasangan, 2–4 set Grad-CAM, grafik accuracy by group, dan paragraf analisis trade-off. Hasil negatif tetap merupakan hasil eksperimen yang sah.

## 9. UX dan Information Architecture Dashboard

Dashboard Streamlit memiliki lima bagian yang sederhana, tanpa autentikasi:

| Tab | Isi dan interaksi |
|---|---|
| **1. Overview** | Penjelasan soal, masalah shortcut, alur baseline → treatment → evaluasi |
| **2. Dataset Explorer** | Empat grup; citra, label, split, frekuensi, contoh foreground–background |
| **3. Model Comparison** | Pilih `sample_id`; dua prediksi berdampingan; ganti latar pada foreground yang sama; tampilkan probabilitas dan benar/salah |
| **4. Evaluation** | Grafik empat grup, overall accuracy, WGA, confusion matrix, ringkasan per seed |
| **5. Explainability & Export** | Grad-CAM untuk contoh yang sama; download CSV/JSON/gambar; penjelasan metode dan keterbatasan |

**Kondisi kosong:** jika belum ada model atau metrik, tampilkan pesan instruksional dan perintah CLI yang relevan, bukan grafik fiktif. Selalu munculkan penanda **SYNTHETIC EXPERIMENT** pada metrik/screenshot prototipe.

## 10. Arsitektur Teknis

```text
                  CONFIG + SEED
                       |
             Synthetic Data Generator
                       |
          Foreground IDs + Background Assets
                       |
                Split by source IDs
                       |
              +--------+---------+
              |                  |
       Biased train set    Balanced train set
              |                  |
        Baseline CNN        Treatment CNN
              |                  |
           checkpoint         checkpoint
              +--------+---------+
                       |
               SAME HELD-OUT TEST
                       |
           Evaluate 4 Groups + Pairs
                       |
        metrics.json / predictions.csv
                       |
           Grad-CAM + Plot Generator
                       |
              Streamlit Dashboard
                       |
             CSV / PNG / PDF report
```

**Prinsip arsitektur:** training dan evaluasi sebagai pipeline offline; Streamlit membaca artefak hasil. Tidak diperlukan FastAPI, database, antrean pekerjaan, atau GPU server untuk MVP. Model di-dashboard adalah model sungguhan dari checkpoint, bukan prediksi hardcoded.

### 10.1 Struktur repo yang ditargetkan

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
│   ├── data/generate.py
│   ├── data/splits.py
│   ├── data/audit.py
│   ├── models/cnn.py
│   ├── training/train.py
│   ├── training/treatment.py
│   ├── evaluation/metrics.py
│   ├── evaluation/counterfactual.py
│   ├── explainability/gradcam.py
│   └── visualization/plots.py
├── scripts/
│   ├── prepare_data.py
│   ├── run_experiment.py
│   └── export_report.py
├── tests/
│   ├── test_dataset.py
│   ├── test_no_leakage.py
│   ├── test_metrics.py
│   └── test_counterfactual.py
├── docs/
│   ├── PRD.md
│   ├── methodology.md
│   └── task-3-report.pdf
├── artifacts/                 # hasil eksperimen; terpisah dari kode
└── notebooks/
    └── demo_colab.ipynb        # opsional
```

**Version control:** source, konfigurasi, dokumentasi, serta grafik ukuran kecil boleh masuk Git; dataset besar, aset dengan lisensi terbatas, dan model checkpoint besar dikecualikan atau disimpan di release/storage yang sesuai. Jaga agar berkas sensitif dan API key tidak ter-commit.

### 10.2 Kontrak artefak minimum

- `manifest.csv`: `image_id,foreground_id,background_asset_id,object_label,background_label,split,seed,file_path`.
- `run_config.json`: seed, versi kode, hyperparameter, rasio bias, waktu eksekusi, model checksum.
- `predictions.csv`: `run_id,model,sample_id,pair_id,y_true,bg,predicted,prob_cow,correct`.
- `metrics.json`: overall, per-group accuracy, worst-group accuracy, flip rate, confusion matrix, seed, data origin (`synthetic`).
- `*.pt`: checkpoint masing-masing model; `*.png`: comparison figure dan Grad-CAM.

**Validasi format:** evaluator harus menghitung metrik dari CSV prediksi dan memverifikasi kesesuaian dengan JSON; file invalid wajib gagal dengan error yang jelas.

## 11. NFR — Non-functional Requirements

| ID | Requirement | Verifikasi |
|---|---|---|
| NFR-01 | Reproducible | Seed/config tersimpan; hasil pada perangkat sekelas mendekati konsisten; bila nondeterminism ada, diungkap |
| NFR-02 | CPU-first | Smoke test, generator, inference, dan dashboard berjalan tanpa GPU; pelatihan GPU hanya opsi |
| NFR-03 | Reliable | Error dataset/checkpoint kosong ditangani; kegagalan tidak ditampilkan sebagai angka berhasil |
| NFR-04 | Testable | Pytest untuk split leakage, rasio grup, WGA, dan pair consistency |
| NFR-05 | Understandable | Reviewer memahami latar masalah dan mekanisme treatment dalam demo 3–5 menit |
| NFR-06 | Transparent | Semua hasil diberi sumber `synthetic`; tak ada klaim foto nyata yang belum diuji |
| NFR-07 | Maintainable | Modul data, model, evaluasi, UI terpisah; konfigurasi tunggal |
| NFR-08 | Responsible assets | Aset foto eksternal hanya dipakai jika sumber/lisensinya jelas dan atribusi disimpan |

## 12. Rencana Implementasi / Ticketing

| Sprint | ID | Ticket | Prioritas | Definition of Done |
|---|---|---|---|---|
| 0 | SET-01 | Inisialisasi repo dan environment | P0 | Instalasi dependency, lint dan pytest berjalan |
| 1 | DAT-01 | Generator cow/non-cow dan grass/beach | P0 | Citra + manifest valid |
| 1 | DAT-02 | Split/source ID dan audit distribusi | P0 | Tidak ada leakage; crosstab valid |
| 2 | ML-01 | CNN + baseline biased training | P0 | Checkpoint, log, config tersedia |
| 2 | ML-02 | Background randomization + treatment | P0 | Grup training seimbang; checkpoint tersedia |
| 3 | EV-01 | Evaluasi empat grup, WGA, counterfactual | P0 | Semua metrik dihitung dari prediksi |
| 3 | EV-02 | Tiga seed + ringkasan komparatif | P0 | mean ± std tersimpan |
| 3 | XAI-01 | Grad-CAM dan ekspor grafik | P0 | Contoh overlay dua model tersedia |
| 4 | UI-01 | Dashboard Streamlit lima tab | P0 | Memuat artefak aktual dan demo berhasil |
| 4 | DOC-01 | README, metodologi, PDF jawaban | P0 | Satu perintah demo, laporan jujur |
| 5 | EXT-01 | Eksperimen foto nyata / Waterbirds | P1/P2 | Dataset resmi/berlisensi dan evaluasi tersendiri |

**Urutan dependensi:** `SET → DAT → ML → EV/XAI → UI → DOC`. Jangan membangun dashboard terlebih dahulu dengan angka statis.

## 13. Quality Gates dan Definition of Done (MVP)

MVP dinyatakan **Done** apabila seluruh pemeriksaan berikut lulus:

1. Menjalankan generator menghasilkan empat grup dan metadata sumber yang lengkap.
2. Train/val/test bebas kebocoran source foreground maupun background; pengecekan otomatis hijau.
3. Rasio latar baseline sesuai konfigurasi bias; treatment independen dari label.
4. Training baseline dan treatment berhasil menyimpan checkpoint untuk sedikitnya tiga seed.
5. Evaluasi pada test set yang sama menghasilkan keseluruhan metrik tanpa hardcode.
6. Counterfactual pair menampilkan objek yang sama dengan dua background berbeda.
7. Grad-CAM dapat dihitung pada checkpoint yang sebenarnya; kelas target jelas.
8. Dashboard berfungsi lokal, termasuk empty state dan ekspor hasil.
9. README berisi instalasi, perintah generate/train/evaluate/demo, keterbatasan, dan sitasi.
10. PDF jawaban menampilkan arsitektur, metode treatment, bukti hasil aktual, dan kesimpulan dengan batasan sintetis.

**Syarat hasil:** tidak diperlukan angka keberhasilan tertentu untuk menganggap pipeline selesai; efektivitas treatment harus disimpulkan sesuai metrik yang benar-benar diobservasi.

## 14. Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Model mempelajari artefak generator, bukan sapi | Kesimpulan menyesatkan | Variasi foreground/warna/pose; audit contoh; uji nyata sebagai lanjutan |
| Foreground sama muncul pada train dan test | Nilai evaluasi terlalu optimistis | Pisahkan source ID sebelum augmentasi dan test otomatis |
| Background asset bocor ke seluruh split | Model menghafal tekstur | Pisahkan keluarga background asset per split |
| Augmentasi membuat artefak tepi terkait kelas | Shortcut baru | Mask/compositing yang konsisten dan inspeksi kualitas |
| Treatment tampak berhasil hanya pada satu seed | Hasil tidak stabil | Tiga seed, laporkan mean dan variasi |
| Grad-CAM ditafsirkan sebagai bukti kausal | Klaim berlebihan | Gunakan bersama per-group metrics dan counterfactual |
| UI memakai angka PDF lama yang di-hardcode | Tidak sah sebagai eksperimen | Semua grafik dari file output pipeline; checksum dan run ID |
| Training memakan sumber daya berlebih | Demo terhambat | CNN ringan, mode smoke-test, run offline terlebih dahulu |
| Penggunaan foto dari internet tanpa izin | Masalah lisensi | Aset sintetis pada MVP; sumber/lisensi wajib untuk real photos |

## 15. Peluncuran, Demonstrasi, dan Handoff

**Definition of release v1.0:** GitHub repo berisi source code, README yang mudah diikuti, contoh artefak kecil yang telah diverifikasi, screenshot dashboard yang sesuai hasil aktual, dan laporan jawaban tugas. Tautan demo publik opsional; demo lokal sepenuhnya diterima sebagai MVP.

**Perintah CLI target (akan diimplementasikan, bukan saat ini tersedia):**

```bash
pip install -e '.[dev]'
python scripts/prepare_data.py --config config/experiment.yaml
python scripts/run_experiment.py --config config/experiment.yaml --seeds 42 123 2026
pytest -q
streamlit run app/streamlit_app.py
```

**Paket handoff:** `README.md`, `docs/PRD.md`, `docs/methodology.md`, `docs/task-3-report.pdf`, konfigurasi eksperimen, manifest dataset, checkpoint (atau tautan unduhan sah), prediksi CSV, metrik JSON, dan figure PNG.

## 16. Roadmap Setelah MVP

**v1.1 — Real-image validation:** dataset foto sapi dan non-sapi berlisensi, mask segmentasi, aturan kualitas komposit, split independen, dan evaluasi terpisah. Jangan menggabungkan skor real dan synthetic tanpa pelabelan jelas.

**v1.2 — Benchmark comparison:** Waterbirds atau benchmark publik sejenis, dengan label foreground–background yang sesuai dan metode pembanding (misalnya group-balanced sampling/Group DRO). Pendekatan ini merupakan perluasan ilmiah, bukan pengganti contoh sapi dari soal.

**v2 — Portfolio polish:** UI lebih editorial, hosting Streamlit, halaman eksperimen publik yang tetap reproducible, CI terukur, dokumentasi berbahasa Inggris, dan contoh failure case.

## 17. Keputusan Produk yang Dikunci

| Pertanyaan | Keputusan |
|---|---|
| Fokus masalah apa? | Fake correlation antara label objek dan latar citra |
| Kasus utama? | Sapi vs bukan sapi; rerumputan vs pantai |
| Model apa? | CNN ringan di PyTorch |
| Treatment utama? | Background randomization dengan sebaran latar seimbang per label |
| Dataset awal? | Sintetis, controlled, repeatable |
| Apa bukti utama? | WGA + empat group accuracy + pasangan counterfactual |
| Apa visualisasi pendukung? | Grad-CAM, confusion matrix, comparison plot |
| UI dan backend? | Streamlit; tanpa backend khusus |
| Apa yang tidak boleh diklaim? | Generalisasi pada foto nyata sebelum pengujian foto nyata |
| Apa tujuan akademiknya? | Menjawab Tugas 3 dengan desain, implementasi treatment, dan visualisasi |

## 18. Referensi Ilmiah

1. Geirhos, R., et al. (2020). *Shortcut learning in deep neural networks*. Nature Machine Intelligence. https://doi.org/10.1038/s42256-020-00257-z
2. Xiao, K. Y., Engstrom, L., Ilyas, A., & Madry, A. (2021). *Noise or Signal: The Role of Image Backgrounds in Object Recognition*. ICLR. https://arxiv.org/abs/2006.09994
3. Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). *Distributionally Robust Neural Networks for Group Shifts*. ICLR. https://arxiv.org/abs/1911.08731
4. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization*. ICCV. https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html

**Status dokumen:** Ini adalah PRD/spesifikasi proyek yang akan dibangun, bukan bukti bahwa repository atau aplikasi sudah diimplementasikan. Angka simulasi yang ada pada PDF jawaban sebelumnya adalah temuan dari prototipe terpisah dan tidak otomatis menjadi angka kinerja repository baru.