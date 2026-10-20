# Panduan presentasi akademik CowBias Lab

Gunakan run asli `stage3-full-38874819983-1` yang sudah disiapkan. Demo tidak menjalankan training
atau inference baru; dashboard membaca bukti tersimpan. Nama, NIM, dan kontribusi individual perlu
dikonfirmasi sebelum pengumpulan. Gunakan kalimat “sistem melakukan...” bila pelaksana individual
belum ditetapkan; jangan mengklaim mahasiswa pribadi menulis atau menjalankan seluruh kode.

## Alur lima menit

| Waktu | Tindakan | Penjelasan inti |
|---|---|---|
| Menit 1: masalah | Overview; sebut sapi/rumput dan sapi/pantai | Korelasi latar dapat membantu menekan loss, tetapi menjadi shortcut yang rapuh ketika konteks berubah. Tujuan tugas adalah menerapkan treatment dan memvisualisasikan perilakunya. |
| Menit 2: data dan treatment | Dataset Explorer; tunjuk pasangan test dengan pair ID sama | Citra 2D sintetis sapi/kuda, rumput/pantai. Baseline 97:3 per kelas, treatment 50:50. Foreground training sama; latar diacak dengan kuota seimbang sebelum training. |
| Menit 3: CNN dan training | Tabel arsitektur dan kurva riwayat pada PDF halaman 5-6 | Tiga blok Conv2D 16/32/64, 89.249 parameter, AdamW, batch 64, tujuh epoch. Dua model dilatih independen dari bobot awal identik. Pilih validation WGA tertinggi, lalu akurasi, lalu epoch awal. Bukan hyperparameter search. |
| Menit 4: hasil | Evaluation, lalu Model Comparison; pilih baseline salah/treatment benar dan toggle Grass/Beach | Periksa sapi/pantai, WGA, flip rate dan paired correctness. Baca probabilitas serta status benar/salah aktual. Seed 42 juga memiliki satu kesalahan baru treatment; jangan menyembunyikannya. |
| Menit 5: penjelasan dan kesimpulan | Explainability & Export; tampilkan peta dan contoh zero map | Grad-CAM memakai kelas benar yang sama. Lokalisasi campuran, bukan bukti kausal tunggal. Treatment memperbaiki robustness terukur dalam eksperimen sintetis; generalisasi foto nyata belum diuji. |

Penutup yang akurat: **“Distribusi latar yang seimbang memperbaiki kinerja kelompok terlemah dan
mengurangi perubahan prediksi akibat pergantian latar dalam eksperimen sintetis ini. Sistem telah
mengimplementasikan treatment dan menyediakan bukti perilaku aktual, dengan batasan yang jelas.”**

## Kontrol demo dan pilihan kasus

Sidebar mengatur run dan seed. Pada ponsel buka sidebar untuk mengganti seed, lalu tutup kembali.
Dataset Explorer, Model Comparison, serta Explainability berbagi pilihan foreground/pair/background.
Filter outcome berlaku bila salah satu anggota pair memenuhi kondisi; anggota yang sesuai dibuka
lebih dahulu. Mengganti latar mempertahankan pair meskipun outcome berubah. Pergantian seed mereset
pilihan yang tidak sesuai. Kategori yang tidak memiliki contoh tetap kosong.

- `Baseline wrong → Treatment correct`: tunjuk koreksi aktual, bukan contoh yang direkayasa.
- `Baseline correct → Treatment wrong`, seed 42: `s42-image-002717`, pair `pair-s42-h-00499`,
  horse/beach; baseline P(cow)=0.484357 benar, treatment P(cow)=0.510905 salah.
- `First baseline zero map`: peta nol asli tetap nol; input ditampilkan tanpa overlay pengganti.
- CSV/JSON/PNG unduhan adalah file ilmiah asli. Overlay pilihan memiliki metadata seed, image,
  target class dan checkpoint. Jangan mengganti hasil full run dengan fixture smoke.

Kasus PDF dipilih secara eksplisit berdasarkan seed lalu image ID terendah dalam masing-masing
kategori koreksi/regresi; hasil pilihan UI bisa berbeda. Ini tidak mengubah metrik seluruh test.
Gambar yang tersedia di dashboard adalah test asli; train/validation hanya ditampilkan sebagai
jumlah dari manifest, bukan klaim adanya piksel train/validation di artefak Stage 3.

## Jawaban singkat untuk pertanyaan dosen

| Pertanyaan | Jawaban yang dapat dipertahankan |
|---|---|
| Mengapa data sintetis? | Agar identitas foreground, latar, proporsi kelompok, mask, dan pemisahan sumber dapat dikontrol dan direproduksi. Konsekuensinya, hasil tidak membuktikan kemampuan pada foto nyata. |
| Mengapa 2D, bukan 3D? | Tugasnya klasifikasi citra dan intervensi korelasi latar. Renderer 2D cukup untuk eksperimen terkendali ini; 3D bukan persyaratan eksplisit dan belum diuji. |
| Mengapa CNN, bukan LLM? | Input berupa citra dengan label biner. CNN menyediakan training, probabilitas, dan atribusi visual yang dapat diperiksa. Soal tidak mewajibkan penggunaan semua DL/LLM/GenAI sekaligus. |
| Bagaimana model dilatih? | Dari nol: input dinormalisasi, forward pass menghasilkan logit, BCEWithLogitsLoss menghitung loss, backpropagation menghitung gradien, AdamW memperbarui parameter. Validation dilakukan setiap epoch tanpa pembaruan bobot. |
| Apakah treatment fine-tuning? | Tidak. Kedua CNN memulai dari bobot awal yang identik, dengan optimizer terpisah. Treatment mengubah distribusi latar training, bukan melanjutkan checkpoint baseline. |
| Apakah ada hyperparameter tuning? | Tidak ada Grid/Random/Bayesian/Optuna search. Ada validation-based checkpoint selection: WGA, akurasi, lalu epoch paling awal. |
| Mengapa tiga seed? | Untuk memperlihatkan variasi dan konsistensi deskriptif. Tiga seed masih sedikit dan bersama-sama memvariasikan data serta inisialisasi; SD bukan interval kepercayaan atau bukti signifikansi. |
| Bagaimana tahu treatment menyebabkan perbedaan? | Desain berpasangan mengendalikan arsitektur, bobot awal, urutan foreground, optimizer, budget, dan validation/test. Assignment latar training adalah perbedaan yang disengaja. Ini mendukung efek intervensi dalam desain tersebut, bukan klaim kausal universal atau bukti model sepenuhnya mengabaikan latar. |
| Bagaimana leakage dicegah? | Source ID/family foreground dan background dialokasikan ke split sebelum rendering; audit juga memeriksa hash sumber, alokasi deterministik, manifest, dan rekontruksi semua komposit. Train dan test tidak berbagi sumber. |
| Mengapa WGA? | Akurasi overall dapat menyembunyikan kelompok buruk. WGA adalah akurasi paling rendah dari sapi/rumput, sapi/pantai, kuda/rumput, dan kuda/pantai. |
| Apakah flip rate rendah cukup? | Tidak. Model selalu menebak satu kelas bisa punya nol flip tetapi salah pada sebagian besar kelompok. Baca bersama paired correctness dan WGA; seed 42 treatment masih memiliki satu pasangan salah pada kedua latar. |
| Apakah Grad-CAM membuktikan ketergantungan foreground? | Tidak. Peta menunjukkan lokalisasi berbasis gradien, target cow=+logit dan horse=-logit. Zero map dapat muncul karena ReLU meski gradien tidak nol. FG share campuran dan kohort nonzero berbeda membatasi interpretasi. |
| Apakah model mengenali sapi sungguhan di pantai? | Belum diketahui. Generalisasi ke foto nyata tidak diuji. Dibutuhkan data foto terpisah dan protokol evaluasi eksternal sebelum membuat klaim itu. |
| Apakah semua hasil treatment sempurna? | Tidak. Ada tujuh prediksi salah secara total, termasuk satu kesalahan baru dibanding baseline. Kasus ini disimpan dan ditampilkan. |
| Apakah arsipnya permanen? | Belum. ZIP asli terverifikasi dan retrieval terdokumentasi; Actions memiliki retensi. Preservasi durabel dengan kontrol akses masih PENDING. |

## Persiapan sebelum presentasi

Ikuti [README](../README.md) untuk install, restore dan menjalankan dashboard. Siapkan PDF sebagai
cadangan demonstrasi offline. Baca [metodologi](methodology.md), [hasil asli](stage3-results.md),
dan [reproducibility](reproducibility.md) agar penjelasan tidak bergantung pada hafalan angka.
Pastikan run/seed terlihat, banner synthetic tidak ditutupi, dan penjelasan keterbatasan disampaikan.
