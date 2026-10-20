"""Indonesian academic PDF, exclusively derived from pinned scientific evidence."""

import argparse
import hashlib
import json
import platform
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from importlib.metadata import version
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from cowbias.evidence import ARCHIVES, PRESERVATION, REPOSITORY, RUN_ID, sha256
from cowbias.report_data import GROUPS, MODELS, SEEDS, load_evidence, read_csv

TITLE = (
    "Perancangan dan Evaluasi Sistem Deep Learning untuk Mengurangi Fake Correlation "
    "pada Klasifikasi Citra Sapi Menggunakan Background Randomization"
)
COMMAND = (
    "python scripts/export_report.py --run-id "
    + RUN_ID
    + " --artifact-root artifacts/evidence/stage3-scientific-results"
    " --training-run artifacts/evidence/stage2-trained-models"
    " --archive-dir artifacts/evidence"
    " --stage4-root artifacts/evidence/stage4-dashboard-review"
    " --output docs/task-3-report.pdf --manifest docs/evidence-manifest.json"
)
COLOR = {"baseline": "#9c4c35", "treatment": "#247770"}


def percent(x):
    return str((Decimal(str(x)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)) + "%"


def summary_stat(data, model, metric):
    return next(
        r for r in data["summary"]["statistics"] if r["model"] == model and r["metric"] == metric
    )


def mean_sd(data, model, metric):
    row = summary_stat(data, model, metric)
    return f"{row['mean'] * 100:.2f} ± {row['sample_sd'] * 100:.2f}"


def plot_save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_figures(data, folder):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    paths = {}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis("off")
    nodes = {
        "a": (5, 5.5, "Generator 2D + alokasi sumber\n48×48 RGB, mask alfa"),
        "b": (5, 4.45, "Split sumber terpisah\ntrain / validation / test"),
        "c": (2.5, 3.3, "Baseline CNN\nlatar 97:3 per kelas"),
        "d": (7.5, 3.3, "Treatment CNN\nlatar 50:50 per kelas"),
        "e": (5, 2.15, "Seleksi checkpoint pada validation\nWGA → akurasi → epoch awal"),
        "f": (5, 1.1, "Evaluasi test identik + counterfactual\nGrad-CAM dari checkpoint beku"),
        "g": (5, 0.05, "Streamlit: lima tab + ekspor bukti"),
    }
    for x, y, label in nodes.values():
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=.45", "fc": "#edf4f3", "ec": "#35706b"},
        )
    for start, end in [
        ("a", "b"),
        ("b", "c"),
        ("b", "d"),
        ("c", "e"),
        ("d", "e"),
        ("e", "f"),
        ("f", "g"),
    ]:
        a, b = nodes[start], nodes[end]
        ax.annotate(
            "",
            xy=(b[0], b[1] + 0.34),
            xytext=(a[0], a[1] - 0.34),
            arrowprops={"arrowstyle": "->", "color": "#596665"},
        )
    paths["pipeline"] = plot_save(fig, folder / "pipeline.png")
    # First lexicographic pair per class; selected without reference to predictions.
    rows = read_csv(data["root"] / "seed_42/manifest.csv")
    examples = []
    for label in ("cow", "non_cow"):
        subset = [r for r in rows if r["split"] == "test" and r["object_label"] == label]
        pair = min(r["pair_id"] for r in subset)
        examples.extend(
            sorted(
                (r for r in subset if r["pair_id"] == pair),
                key=lambda r: r["background_label"] == "beach",
            )
        )
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.7))
    ids = data["arrays"][42]["image_ids"].tolist()
    for ax, row in zip(axes, examples, strict=True):
        ax.imshow(
            data["arrays"][42]["source_images"][ids.index(row["image_id"])], interpolation="nearest"
        )
        label = "sapi" if row["object_label"] == "cow" else "kuda"
        ax.set_title(f"{label} / {row['background_label']}", fontsize=10)
        ax.set_xlabel(row["image_id"] + "\n" + row["pair_id"], fontsize=7)
        ax.set(xticks=[], yticks=[])
    fig.tight_layout()
    paths["examples"] = plot_save(fig, folder / "examples.png")
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.1), sharex=True)
    for col, seed in enumerate(SEEDS):
        for model in MODELS:
            history = data["histories"][seed, model]
            epochs = [int(r["epoch"]) for r in history]
            chosen = data["metadata"][seed, model]["selected_epoch"]
            for row, (key, scale) in enumerate([("train_loss", 1), ("val_wga", 100)]):
                values = [float(r[key]) * scale for r in history]
                axes[row, col].plot(
                    epochs, values, color=COLOR[model], label=model, marker="o", markersize=3
                )
                axes[row, col].scatter(
                    [chosen],
                    [values[chosen - 1]],
                    marker="*",
                    s=120,
                    color=COLOR[model],
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=4,
                )
                axes[row, col].grid(alpha=0.18)
                axes[row, col].set_xticks(epochs)
            axes[1, col].set(ylim=(-3, 103), xlabel="Epoch")
        axes[0, col].set_title(f"Seed {seed}")
    axes[0, 0].set_ylabel("Training loss (BCE)")
    axes[1, 0].set_ylabel("Validation WGA (%)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    paths["histories"] = plot_save(fig, folder / "histories.png")
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.7))
    labels = ["S/R", "S/P", "K/R", "K/P"]
    for ax, seed in zip(axes, SEEDS, strict=True):
        for j, model in enumerate(MODELS):
            values = [
                data["metrics"][seed]["models"][model]["groups"][g]["accuracy"] * 100
                for g in GROUPS
            ]
            ax.bar(
                np.arange(4) + (j - 0.5) * 0.35, values, width=0.35, label=model, color=COLOR[model]
            )
        ax.set(xticks=np.arange(4), xticklabels=labels, ylim=(0, 105), title=f"Seed {seed}")
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Akurasi (%)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    paths["groups"] = plot_save(fig, folder / "groups.png")
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.2))
    for col, seed in enumerate(SEEDS):
        for row, model in enumerate(MODELS):
            ax = axes[row, col]
            matrix = np.asarray(data["metrics"][seed]["models"][model]["confusion_matrix"])
            ax.imshow(matrix, vmin=0, vmax=240, cmap="Blues")
            for (y, x), count in np.ndenumerate(matrix):
                ax.text(
                    x,
                    y,
                    str(count),
                    ha="center",
                    va="center",
                    color="white" if count > 120 else "black",
                    fontsize=10,
                )
            ax.set(
                xticks=[0, 1],
                yticks=[0, 1],
                xticklabels=["Kuda", "Sapi"],
                yticklabels=["Kuda", "Sapi"],
                title=f"{seed} / {model}",
            )
            if row == 1:
                ax.set_xlabel("Prediksi")
            if col == 0:
                ax.set_ylabel("Aktual")
    fig.tight_layout()
    paths["confusions"] = plot_save(fig, folder / "confusions.png")
    case_images = []
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.8))
    for row, case in enumerate(data["cases"]):
        seed = case["seed"]
        pair_rows = [
            r
            for (m, _), r in data["predictions"][seed].items()
            if m == "baseline" and r["pair_id"] == case["pair_id"]
        ]
        pair_rows.sort(key=lambda r: r["background_label"] == "beach")
        for ax, item in zip(axes[row], pair_rows, strict=True):
            idx = data["arrays"][seed]["image_ids"].tolist().index(item["image_id"])
            ax.imshow(data["arrays"][seed]["source_images"][idx], interpolation="nearest")
            ax.set_title(f"Seed {seed} / {item['background_label']}", fontsize=9)
            ax.set_xlabel(item["image_id"] + "\n" + item["pair_id"], fontsize=7)
            ax.set(xticks=[], yticks=[])
            case_images.append(
                {"seed": seed, "image_id": item["image_id"], "pair_id": item["pair_id"]}
            )
    fig.tight_layout()
    paths["cases"] = plot_save(fig, folder / "cases.png")
    # Genuine saved heatmaps on the two disclosed observed case inputs.
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.6))
    for row, case in enumerate(data["cases"]):
        seed, image = case["seed"], case["image_id"]
        idx = data["arrays"][seed]["image_ids"].tolist().index(image)
        original = data["arrays"][seed]["images"][idx]
        axes[row, 0].imshow(original, interpolation="nearest")
        axes[row, 0].set_title(f"Input 32×32 / {case['object_label']}", fontsize=9)
        for col, model in enumerate(MODELS, 1):
            with np.load(
                data["root"] / f"seed_{seed}/{model}_heatmaps.npz", allow_pickle=False
            ) as n:
                heat = n["heatmaps"][idx]
            axes[row, col].imshow(original, interpolation="nearest")
            if heat.any():
                axes[row, col].imshow(heat, vmin=0, vmax=1, cmap="inferno", alpha=0.5)
            axes[row, col].set_title(model + (" (zero map)" if not heat.any() else ""), fontsize=9)
        for ax in axes[row]:
            ax.set(xticks=[], yticks=[])
            ax.set_xlabel(f"s{seed}: {image}", fontsize=7)
    fig.tight_layout()
    paths["gradcam"] = plot_save(fig, folder / "gradcam.png")
    return paths, examples, case_images


class AcademicPDF:
    def __init__(self, output, data, folder):
        self.data, self.folder = data, folder
        self.story, self.entries, self.page = [], [], 1
        self.styles = {
            "body": ParagraphStyle(
                "body",
                fontName="Times-Roman",
                fontSize=11,
                leading=15,
                alignment=TA_JUSTIFY,
                spaceAfter=8,
            ),
            "h1": ParagraphStyle(
                "h1",
                fontName="Times-Bold",
                fontSize=16,
                leading=20,
                textColor=colors.HexColor("#164c49"),
                spaceAfter=12,
            ),
            "h2": ParagraphStyle(
                "h2", fontName="Times-Bold", fontSize=12, leading=16, spaceBefore=6, spaceAfter=6
            ),
            "caption": ParagraphStyle(
                "caption", fontName="Times-Roman", fontSize=9, leading=12, spaceAfter=9
            ),
            "cell": ParagraphStyle("cell", fontName="Times-Roman", fontSize=9, leading=11),
            "cover": ParagraphStyle(
                "cover",
                fontName="Times-Bold",
                fontSize=22,
                leading=28,
                alignment=TA_CENTER,
                spaceAfter=20,
            ),
            "center": ParagraphStyle(
                "center",
                fontName="Times-Roman",
                fontSize=12,
                leading=18,
                alignment=TA_CENTER,
                spaceAfter=14,
            ),
        }
        self.doc = SimpleDocTemplate(
            str(output),
            pagesize=A4,
            rightMargin=20 * mm,
            leftMargin=20 * mm,
            topMargin=19 * mm,
            bottomMargin=18 * mm,
            title=TITLE,
            author="CowBias Lab",
            invariant=1,
        )

    def p(self, text, style="body"):
        self.story.append(Paragraph(text, self.styles[style]))

    def newpage(self):
        self.story.append(PageBreak())
        self.page += 1

    def source(self, stage, relative):
        bases = {
            "stage2-trained-models": self.data["training"],
            "stage3-scientific-results": self.data["root"],
            "stage4-dashboard-review": self.data["stage4"],
            "repository": Path.cwd(),
        }
        path = bases[stage] / relative
        return {
            "artifact": stage,
            "internal_path": (RUN_ID + "/" if stage == "stage3-scientific-results" else "")
            + relative,
            "sha256": sha256(path),
            "source_run_id": "stage2-full"
            if stage.startswith("stage2")
            else RUN_ID
            if stage.startswith(("stage3", "stage4"))
            else None,
        }

    def trace(self, identifier, section, sources, **extra):
        self.entries.append(
            {
                "id": identifier,
                "report_page": self.page,
                "section": section,
                "sources": sources,
                "generation_command": COMMAND,
                "generation_script": "src/cowbias/report.py",
                **extra,
            }
        )

    def table(self, identifier, caption, cells, widths, sources, section):
        self.p(f"<b>{identifier}.</b> {caption}", "caption")
        formatted = [[Paragraph(escape(str(x)), self.styles["cell"]) for x in row] for row in cells]
        table = Table(formatted, colWidths=[x * mm for x in widths], repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5efed")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#45716e")),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#718784")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f6f8f7")],
                    ),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        self.story.extend([table, Spacer(1, 8)])
        self.trace(identifier, section, sources, cells=cells, kind="derived_table")

    def figure(self, identifier, caption, path, sources, section, max_height=90, **extra):
        with PILImage.open(path) as im:
            w, h = im.size
            pixels = hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
        factor = min(170 * mm / w, max_height * mm / h)
        self.story.append(Image(str(path), width=w * factor, height=h * factor))
        self.p(f"<b>{identifier}.</b> {caption}", "caption")
        self.trace(
            identifier,
            section,
            sources,
            kind="figure",
            figure_sha256=sha256(path),
            rgb_pixels_sha256=pixels,
            pixel_dimensions=[w, h],
            **extra,
        )

    def build(self):
        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont("Times-Roman", 9)
            canvas.setFillColor(colors.HexColor("#566664"))
            canvas.drawString(
                20 * mm, 10 * mm, "CowBias Lab | Tugas 3 Kecerdasan Buatan | Eksperimen sintetis"
            )
            canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, str(doc.page))
            canvas.restoreState()

        self.doc.build(self.story, onFirstPage=footer, onLaterPages=footer)


def compose(data, output, folder):
    figs, examples, case_images = make_figures(data, folder)
    doc = AcademicPDF(output, data, folder)
    s2 = lambda p: doc.source("stage2-trained-models", p)
    s3 = lambda p: doc.source("stage3-scientific-results", p)
    repo = lambda p: doc.source("repository", p)
    nparams = data["metadata"][42, "baseline"]["parameter_count"]
    settings = data["config"]["effective_config"]["training"]
    # Page 1: identification remains deliberately unassigned.
    doc.story.append(Spacer(1, 28 * mm))
    doc.p("LAPORAN TUGAS 3", "center")
    doc.p(TITLE, "cover")
    doc.p(
        "Artificial Intelligence / Kecerdasan Buatan<br/>Magister Informatika<br/>"
        "UIN Sunan Kalijaga",
        "center",
    )
    doc.story.append(Spacer(1, 15 * mm))
    doc.p(
        "Nama mahasiswa: [diisi sebelum pengumpulan]<br/>"
        "NIM: [diisi sebelum pengumpulan]<br/>Kelas: [diisi sebelum pengumpulan]",
        "center",
    )
    doc.story.append(Spacer(1, 15 * mm))
    doc.p("CowBias Lab - laporan akademik Stage 5<br/>Versi 19 Oktober 2026", "center")
    doc.p(
        "Laporan berdasarkan pelatihan CNN dan evaluasi sintetis asli yang tersimpan. "
        "Identitas, kontribusi individual, dan ketentuan penggunaan bantuan perangkat lunak/AI "
        "perlu dikonfirmasi oleh mahasiswa sesuai aturan perkuliahan. Laporan ini tidak "
        "menetapkan klaim kepengarangan individual atau pengesahan institusi.",
        "caption",
    )
    doc.p(
        f'<link href="https://github.com/{REPOSITORY}" color="#176b66">'
        f"https://github.com/{REPOSITORY}</link>",
        "center",
    )
    # Page 2.
    doc.newpage()
    doc.p("1. Pendahuluan", "h1")
    doc.p(
        "Fake correlation atau korelasi semu adalah hubungan prediktif pada data yang tidak "
        "mewakili ciri objek yang stabil ketika konteks berubah. CNN dapat menekan loss dengan "
        "memanfaatkan warna, tekstur, dan pola latar yang sering muncul bersama label. Jika sapi "
        "hampir selalu berada di rumput, fitur rumput dapat menjadi shortcut. Keberhasilan pada "
        "distribusi tersebut tidak menjamin keberhasilan untuk sapi di pantai [1,2]."
    )
    doc.p(
        "Instruksi Tugas 3 tanggal 14 Oktober 2026, sebagaimana disediakan pada permintaan "
        "proyek, berbunyi:"
    )
    doc.p(
        "“Mendesain sistem yang mengimplementasikan perbaikan pada fake correlation. Sebagai "
        "contoh, sistem mengenali sapi pada gambar-gambar yang memiliki latar belakang "
        "rerumputan. Ketika diberikan gambar sapi di pantai, sistem bisa salah mengenali sapi. "
        "Sistem yang diimplementasikan bisa menangani hal tersebut.”"
    )
    doc.p(
        "Versi Inggris menegaskan: “Students should hand in a visualization model with an "
        "explanation of the treatment on system.” Dokumen DOCX asli tidak tersedia pada "
        "workspace Stage 5; kutipan di atas berasal dari teks yang diberikan pengguna, "
        "dengan konteks desain dari PRD repository. Tidak ditambahkan rubrik atau tenggat tugas."
    )
    doc.p("2. Rumusan Masalah dan Tujuan", "h1")
    doc.p(
        "Pertanyaan penelitian adalah apakah perubahan distribusi latar pada data latih "
        "mengurangi ketergantungan prediksi pada latar serta meningkatkan akurasi kelompok "
        "terlemah pada gambar held-out yang identik. Hipotesisnya: latar yang seimbang untuk "
        "setiap label mengurangi kegunaan latar sebagai petunjuk label. Hipotesis ini diuji "
        "melalui dua CNN dan tidak dianggap benar sebelum hasil diamati."
    )
    doc.table(
        "Tabel 1",
        "Pemisahan kebutuhan tugas dan keputusan proyek.",
        [
            ["Kategori", "Ruang lingkup"],
            [
                "Instruksi eksplisit",
                "Merancang sistem, menerapkan treatment korelasi semu, menyajikan visualisasi dan penjelasan treatment.",
            ],
            [
                "Keputusan rekayasa",
                "Data sintetis 2D, kuda sebagai pembanding, CNN, tiga seed, WGA, Grad-CAM, Streamlit, GitHub, dan laporan terprogram.",
            ],
            [
                "Temuan ilmiah",
                "Angka hanya berasal dari checkpoint dan artefak full run yang diverifikasi; hasil dibahas pada Bagian 8-10.",
            ],
            [
                "Batas klaim",
                "Tidak menguji foto nyata, tidak membuktikan independensi kausal universal dari latar.",
            ],
        ],
        [37, 133],
        [repo("docs/PRD.md")],
        "1-2",
    )
    doc.p(
        "Sub-CPMK menyebut Deep Learning, LLM, dan Generative AI, tetapi instruksi spesifik "
        "tidak mewajibkan ketiganya dalam satu sistem. CNN dipilih karena masalahnya adalah "
        "klasifikasi citra biner dengan intervensi distribusi yang dapat dikendalikan."
    )
    # Page 3.
    doc.newpage()
    doc.p("3. Desain Sistem", "h1")
    doc.p(
        "Sistem memisahkan pembangkitan data, optimasi parameter, seleksi checkpoint, dan "
        "evaluasi. Test tidak digunakan untuk pembaruan parameter atau pemilihan checkpoint. "
        "Dashboard hanya membaca hasil tersimpan; interaksi pengguna tidak melatih model."
    )
    doc.figure(
        "Gambar 1",
        "Alur implementasi yang diturunkan dari modul sumber; diagram "
        "penjelas, bukan hasil pengukuran. Berlaku untuk seed 42, 123, dan 2026.",
        figs["pipeline"],
        [
            repo("src/cowbias/data/pipeline.py"),
            repo("src/cowbias/training/experiment.py"),
            repo("src/cowbias/evaluation/evaluator.py"),
            repo("src/cowbias/dashboard/ui.py"),
        ],
        "3",
        max_height=90,
        evidence_type="source-derived explanatory diagram",
        seeds=list(SEEDS),
    )
    doc.p("4. Dataset", "h1")
    doc.p(
        "Data berupa ilustrasi 2D yang dibangkitkan secara prosedural menggunakan NumPy dan "
        "Pillow. Dua label objek adalah sapi (cow=1) dan kuda sintetis (non_cow=0); dua latar "
        "adalah rumput (grass) dan pantai (beach). Gambar sumber RGB berukuran 48×48, lalu "
        "diubah menjadi RGB 32×32 untuk CNN. Tidak ada foto stok atau citra generatif pengganti."
    )
    doc.figure(
        "Gambar 2",
        "Contoh asli test_inputs.npz, seed 42. Pasangan pertama secara "
        "leksikografis untuk setiap kelas dipilih tanpa melihat keberhasilan prediksi. "
        "Setiap pasangan memakai foreground identik pada rumput dan pantai.",
        figs["examples"],
        [s3("seed_42/test_inputs.npz"), s3("seed_42/manifest.csv")],
        "4",
        max_height=56,
        seed=42,
        images=[{k: r[k] for k in ("image_id", "pair_id", "foreground_id")} for r in examples],
    )
    # Page 4.
    doc.newpage()
    doc.p("4. Dataset (lanjutan)", "h1")
    distributions = data["distributions"][42]
    cells = [
        [
            "Split / varian",
            "Sapi / rumput",
            "Sapi / pantai",
            "Kuda / rumput",
            "Kuda / pantai",
            "Total",
        ]
    ]
    for name in ("baseline", "treatment", "validation", "test"):
        counts = [distributions[name][g] for g in GROUPS]
        cells.append([name, *counts, sum(counts)])
    doc.table(
        "Tabel 2",
        "Jumlah komposit per seed dari manifest audit. Nilai identik untuk "
        "seed 42, 123, 2026; train baseline dan treatment adalah dua versi foreground yang sama.",
        cells,
        [35, 27, 27, 27, 27, 27],
        [s3(f"seed_{s}/manifest.csv") for s in SEEDS],
        "4",
    )
    doc.p(
        "Setiap seed memiliki 1.000 foreground latih yang digunakan ulang oleh kedua arm. "
        "Validation berisi 240 foreground terpisah. Test terdiri atas 240 foreground unik, "
        "masing-masing dirender di dua latar sehingga menghasilkan 480 gambar. Kedua model "
        "menggunakan validation dan test yang sama. Gambar pasangan bukan observasi hewan "
        "independen; satu identitas foreground muncul dua kali dalam test."
    )
    doc.p("4.1 Isolasi sumber dan determinisme", "h2")
    doc.p(
        "Identitas dan family foreground serta latar dialokasikan ke train/validation/test "
        "sebelum rendering. Turunan sumber harus mengikuti split asalnya. Audit memeriksa "
        "overlap ID, family, dan hash sumber, termasuk sumber latar yang tidak terpakai. "
        "Distribusi renderer, skala, penempatan, orientasi, dan variasi warna digunakan bersama "
        "antarspesies; anatomi dan pola tubuh membedakan sapi dari kuda."
    )
    doc.p(
        "Random stream ditentukan oleh seed dan source ID, bukan hash acak proses Python. "
        "Audit mengulang alokasi sumber dan penetapan latar, memeriksa manifest, serta "
        "merekonstruksi setiap komposit dari sumber. Ketiga audit asli berstatus PASS. "
        "Status ini menjelaskan pemeriksaan yang telah dilakukan, bukan jaminan bahwa "
        "generator bebas dari semua petunjuk tidak disengaja."
    )
    doc.p("4.2 Mask dan compositing", "h2")
    doc.p(
        "Foreground RGBA digabungkan dengan latar RGB melalui alpha compositing: "
        "I = alpha × F + (1 - alpha) × B. Mask menyimpan alfa foreground. Pada pasangan "
        "counterfactual, mask dan piksel foreground opak tetap identik; piksel tepi dengan "
        "alfa parsial dapat berubah karena berbaur dengan latar baru. Karena itu, treatment "
        "tidak diartikan sebagai perubahan hanya pada piksel yang sepenuhnya di luar objek."
    )
    doc.p(
        "Distribusi empat kelompok memungkinkan kegagalan sapi/pantai dan kuda/rumput "
        "terlihat, walaupun rerata akurasi tampak cukup tinggi. Angka train/validation berasal "
        "dari manifest; gambar contoh laporan adalah piksel test asli dari artefak Stage 3."
    )
    # Page 5.
    doc.newpage()
    doc.p("5. Model CNN dan Proses Training", "h1")
    doc.table(
        "Tabel 3",
        f"SmallCNN bersama dengan {nparams:,} parameter trainable.",
        [
            ["Komponen", "Operasi / dimensi"],
            ["Input", "RGB 32×32; tensor float32 CHW; (pixel/255 - 0.5)/0.5"],
            ["Blok 1", "Conv2D 3→16, kernel 3×3, padding 1; ReLU; MaxPool2D(2)"],
            ["Blok 2", "Conv2D 16→32, kernel 3×3, padding 1; ReLU; MaxPool2D(2)"],
            ["Blok 3", "Conv2D 32→64, kernel 3×3, padding 1; ReLU; MaxPool2D(2)"],
            [
                "Classifier",
                "AdaptiveAvgPool2D(4×4); Flatten(1024); Linear(1024,64); ReLU; Linear(64,1)",
            ],
            [
                "Optimasi",
                f"AdamW; learning rate {settings['learning_rate']}; batch {settings['batch_size']}; {settings['epochs']} epoch; BCEWithLogitsLoss",
            ],
        ],
        [32, 138],
        [repo("src/cowbias/models/cnn.py"), s2("seed_42/baseline_metadata.json")],
        "5",
    )
    doc.p(
        "Resize memakai bilinear Pillow, tanpa augmentasi tambahan. Model dilatih dari nol. "
        "Forward pass menghasilkan satu logit z; BCEWithLogitsLoss membandingkan logit dengan "
        "label biner. Gradien dihitung melalui backpropagation, lalu optimizer.step() memperbarui "
        "bobot setelah gradien batch sebelumnya dibersihkan. Validation dilakukan tanpa gradien "
        "setelah setiap epoch. AdamW memakai weight decay 0.01 sesuai metadata asli."
    )
    doc.p(
        "Untuk setiap seed, dua model dimulai dari salinan parameter awal identik, tetapi "
        "memiliki optimizer dan proses optimasi terpisah. Hash inisialisasi, urutan foreground "
        "setiap epoch, batch size, learning rate, budget tujuh epoch, preprocessing, dan "
        "validation set cocok. Bobot akhir dapat berbeda karena latar training berbeda."
    )
    doc.p("6. Mekanisme Treatment", "h1")
    doc.p(
        "Baseline memasangkan 97% sapi dengan rumput dan 97% kuda dengan pantai: "
        "485/15/15/485 gambar per kelompok. Treatment mengacak penetapan latar dengan kuota "
        "tepat 50:50 untuk setiap label: 250/250/250/250. Jumlah label dan foreground tidak "
        "ditambah. Penetapan latar dilakukan sebelum training, dengan urutan foreground yang sama."
    )
    doc.p(
        "Dalam data seimbang, kategori latar saja tidak lagi membedakan label secara sistematis. "
        "Model perlu memanfaatkan petunjuk lain untuk mengurangi loss. Namun, seimbangnya "
        "frekuensi tidak menjamin model hanya memakai bentuk hewan; ketergantungan tersisa "
        "harus diperiksa melalui test dan counterfactual. Treatment bukan fine-tuning checkpoint "
        "baseline dan bukan Grad-CAM. Tidak ada perubahan threshold setelah hasil test dilihat."
    )
    # Page 6.
    doc.newpage()
    doc.p("7. Training History dan Model Selection", "h1")
    doc.p(
        "Gambar berikut berasal dari enam CSV riwayat training asli Stage 2. Tanda bintang "
        "menunjukkan checkpoint yang dipilih, bukan penghentian training. Semua model tetap "
        "menyelesaikan tujuh epoch. Variasi validation WGA memperlihatkan mengapa epoch "
        "terakhir atau training loss terendah tidak otomatis dipilih."
    )
    history_sources = [s2(f"seed_{s}/{m}_history.csv") for s in SEEDS for m in MODELS]
    doc.figure(
        "Gambar 3",
        "Training loss dan validation WGA untuk seluruh seed 42, 123, 2026. "
        "Sumber: baseline_history.csv dan treatment_history.csv Stage 2; bintang = epoch terpilih.",
        figs["histories"],
        history_sources,
        "7",
        max_height=104,
        seeds=list(SEEDS),
    )
    cells = [["Seed", "Epoch baseline", "Epoch treatment", "Val WGA baseline", "Val WGA treatment"]]
    for s in SEEDS:
        a, b = (data["metadata"][s, m] for m in MODELS)
        cells.append(
            [
                s,
                a["selected_epoch"],
                b["selected_epoch"],
                percent(a["selected_validation"]["worst_group_accuracy"]),
                percent(b["selected_validation"]["worst_group_accuracy"]),
            ]
        )
    doc.table(
        "Tabel 4",
        "Checkpoint dipilih hanya menggunakan validation.",
        cells,
        [20, 35, 35, 40, 40],
        history_sources + [s2(f"seed_{s}/{m}_metadata.json") for s in SEEDS for m in MODELS],
        "7",
    )
    doc.p(
        "Aturan seleksi berurutan: (1) validation worst-group accuracy tertinggi; "
        "(2) validation overall accuracy tertinggi sebagai tie-breaker; (3) epoch paling awal "
        "bila kedua skor sama. Epoch dan skor diverifikasi ulang terhadap riwayat dan metadata "
        "checkpoint. Test tidak dilibatkan dalam seleksi."
    )
    doc.p(
        "Tidak dilakukan Grid Search, Random Search, Bayesian Optimization, atau Optuna. "
        "Konfigurasi training ditetapkan untuk eksperimen terkontrol. Pemilihan checkpoint "
        "berbasis validation adalah seleksi model sepanjang training, bukan kampanye "
        "optimasi hyperparameter yang luas."
    )
    # Page 7.
    doc.newpage()
    doc.p("8. Hasil Evaluasi", "h1")
    doc.p(
        "Setiap seed menggunakan 480 gambar test yang sama untuk kedua checkpoint, dengan "
        "120 gambar per kelompok. Probabilitas sapi adalah sigmoid(z); prediksi sapi bila "
        "z ≥ 0, ekuivalen dengan P(sapi) ≥ 0.5. Akurasi overall adalah proporsi benar; "
        "WGA adalah minimum akurasi empat kelompok. Flip rate menghitung pasangan yang "
        "prediksinya berubah; paired correctness menghitung pasangan yang keduanya benar."
    )
    metric_labels = [
        ("overall_accuracy", "Overall accuracy"),
        ("worst_group_accuracy", "Worst-group accuracy"),
        ("background_flip_rate", "Background flip rate"),
        ("paired_correct_rate", "Paired correctness"),
        ("group_accuracy/cow/beach", "Akurasi sapi/pantai"),
        ("group_accuracy/cow/grass", "Akurasi sapi/rumput"),
        ("group_accuracy/non_cow/grass", "Akurasi kuda/rumput"),
        ("group_accuracy/non_cow/beach", "Akurasi kuda/pantai"),
    ]
    cells = [["Metrik (%)", "Baseline mean ± SD", "Treatment mean ± SD"]]
    for metric, label in metric_labels:
        cells.append([label, *[mean_sd(data, m, metric) for m in MODELS]])
    doc.table(
        "Tabel 5",
        "Rerata dan simpangan baku sampel (ddof=1) atas tiga seed. "
        "SD bersifat deskriptif; bukan interval kepercayaan.",
        cells,
        [58, 56, 56],
        [s3("summary.json"), s3("summary.csv")],
        "8",
    )
    cells = [["Seed / model", "Overall", "WGA", "Flip", "Pasangan benar"]]
    for s in SEEDS:
        for m in MODELS:
            r = data["metrics"][s]["models"][m]
            cf = r["counterfactual"]
            cells.append(
                [
                    f"{s} / {m}",
                    percent(r["overall_accuracy"]),
                    percent(r["worst_group_accuracy"]),
                    percent(cf["background_flip_rate"]),
                    f"{cf['both_correct_pairs']}/{cf['total_pairs']} ({percent(cf['paired_correct_rate'])})",
                ]
            )
    doc.table(
        "Tabel 6",
        "Hasil per seed; denominator flip dan paired correctness = 240 pasangan.",
        cells,
        [40, 27, 27, 27, 49],
        [s3(f"seed_{s}/metrics.json") for s in SEEDS],
        "8",
    )
    a = summary_stat(data, "baseline", "group_accuracy/cow/beach")["mean"]
    b = summary_stat(data, "treatment", "group_accuracy/cow/beach")["mean"]
    doc.p(
        f"Kasus inti tugas, sapi di pantai, meningkat dari rerata {percent(a)} menjadi "
        f"{percent(b)}. Namun baseline sangat bervariasi antar-seed. Peningkatan WGA dan "
        "paired correctness terjadi pada ketiga seed; klaim ini dibatasi oleh generator dan "
        "protokol uji yang digunakan. Rerata tiga seed bukan hasil satu model gabungan."
    )
    doc.p(
        "Akurasi kelompok terlemah penting karena akurasi agregat dapat menyamarkan "
        "kegagalan pada kombinasi objek-latar tertentu [3]. Proyek ini memakai WGA sebagai "
        "metrik dan seleksi checkpoint; algoritma Group DRO dari referensi [3] tidak diterapkan."
    )
    # Page 8.
    doc.newpage()
    doc.p("8. Hasil Evaluasi (lanjutan)", "h1")
    cells = [["Seed / model", "Sapi / rumput", "Sapi / pantai", "Kuda / rumput", "Kuda / pantai"]]
    for s in SEEDS:
        for m in MODELS:
            cells.append(
                [
                    f"{s} / {m}",
                    *[
                        f"{data['metrics'][s]['models'][m]['groups'][g]['correct']}/120"
                        for g in GROUPS
                    ],
                ]
            )
    sources = [s3(f"seed_{s}/metrics.json") for s in SEEDS]
    doc.table(
        "Tabel 7",
        "Jumlah prediksi benar per kelompok. Denominator setiap sel = 120 gambar.",
        cells,
        [42, 32, 32, 32, 32],
        sources,
        "8",
    )
    doc.figure(
        "Gambar 4",
        "Akurasi kelompok dari metrics.json, seed 42/123/2026. "
        "S=sapi, K=kuda, R=rumput, P=pantai. Sumbu konsisten 0-100%.",
        figs["groups"],
        sources,
        "8",
        max_height=53,
        seeds=list(SEEDS),
    )
    doc.figure(
        "Gambar 5",
        "Enam confusion matrix asli dihitung dari prediksi yang sama. "
        "Baris=kelas aktual, kolom=prediksi; urutan kuda lalu sapi, skala warna 0-240. "
        "Sumber: metrics.json dan predictions.csv, seed 42/123/2026.",
        figs["confusions"],
        sources + [s3(f"seed_{s}/predictions.csv") for s in SEEDS],
        "8",
        max_height=85,
        seeds=list(SEEDS),
    )
    corrected = sum(data["metrics"][s]["comparison"]["treatment_corrected"] for s in SEEDS)
    new = sum(data["metrics"][s]["comparison"]["new_errors"] for s in SEEDS)
    residual = sum(
        data["metrics"][s]["models"]["treatment"]["total"]
        - data["metrics"][s]["models"]["treatment"]["correct"]
        for s in SEEDS
    )
    doc.p(
        f"Secara deskriptif, treatment memperbaiki {corrected} kesalahan baseline dan "
        f"menimbulkan {new} kesalahan baru; {residual} prediksi treatment tetap salah. "
        "Penjumlahan ini melintasi seed dan pasangan yang bergantung, sehingga tidak "
        "dipakai sebagai estimasi ketidakpastian dari observasi independen."
    )
    # Page 9.
    doc.newpage()
    doc.p("9. Counterfactual Analysis", "h1")
    doc.p(
        "Dua contoh eksploratif dipilih dengan aturan tetap: urutan seed lalu image ID "
        "terkecil dalam kategori baseline salah/treatment benar dan baseline benar/treatment "
        "salah. Kedua anggota setiap pasangan ditampilkan. Pemilihan berdasarkan outcome "
        "ini menjelaskan perilaku dan tidak mengestimasi prevalensi; semua kasus tetap "
        "tercakup dalam metrik agregat."
    )
    doc.figure(
        "Gambar 6",
        "Pasangan asli test_inputs.npz, seed 42. Baris pertama: koreksi "
        "baseline. Baris kedua: kesalahan baru treatment. Image ID dan pair ID "
        "memastikan foreground dipertahankan; tepi alfa lunak dapat berubah.",
        figs["cases"],
        [s3("seed_42/test_inputs.npz"), s3("seed_42/model_disagreements.csv")],
        "9",
        max_height=90,
        images=case_images,
        seed=42,
    )
    cells = [
        ["Image ID / aktual", "Latar", "Baseline prediksi; P(sapi)", "Treatment prediksi; P(sapi)"]
    ]
    for info in case_images:
        seed, image = info["seed"], info["image_id"]
        a, b = (data["predictions"][seed][m, image] for m in MODELS)
        truth = "sapi" if a["y_true"] == "1" else "kuda"

        def cell(r):
            label = "sapi" if r["predicted"] == "1" else "kuda"
            correct = "benar" if r["correct"] == "1" else "salah"
            return f"{label}; {float(r['prob_cow']):.6f} ({correct})"

        cells.append([f"{image} / {truth}", a["background_label"], cell(a), cell(b)])
    doc.table(
        "Tabel 8",
        "Probabilitas, kelas, dan kebenaran dari baris prediksi asli; threshold 0.5.",
        cells,
        [53, 22, 47.5, 47.5],
        [s3("seed_42/predictions.csv"), s3("seed_42/counterfactual_pairs.csv")],
        "9",
    )
    doc.p(
        "Low flip rate tidak cukup: model yang selalu menebak satu kelas dapat memiliki "
        "nol flip sekaligus salah pada banyak objek. Dalam hasil asli, treatment seed 42 "
        "masih memiliki satu pasangan yang salah pada kedua latar. Oleh karena itu, "
        "stabilitas selalu dibaca bersama paired correctness dan WGA."
    )
    # Page 10.
    doc.newpage()
    doc.p("10. Grad-CAM dan Visualisasi", "h1")
    doc.p(
        "Grad-CAM memakai gradien skor kelas benar pada aktivasi setelah Conv2D terakhir "
        "(features.7): +logit untuk sapi dan -logit untuk kuda, sama bagi kedua model [4]. "
        "Parameter dibekukan; fingerprint state sebelum/sesudah cocok. Peta memakai ReLU, "
        "upsampling bilinear dan normalisasi puncak positif per citra."
    )
    doc.figure(
        "Gambar 7",
        "Input model dan overlay peta Grad-CAM asli untuk dua kasus "
        "Gambar 6, seed 42. Target kelas benar identik antar-model; inferno, alfa 0.50. "
        "Zero map mempertahankan input tanpa heatmap pengganti.",
        figs["gradcam"],
        [
            s3("seed_42/test_inputs.npz"),
            s3("seed_42/baseline_heatmaps.npz"),
            s3("seed_42/treatment_heatmaps.npz"),
            s3("seed_42/attribution.csv"),
        ],
        "10",
        max_height=74,
        seed=42,
        images=data["cases"],
        target="true_class",
    )
    cells = [["Seed", "Zero B / T", "FG share B", "FG share T"]]
    for s in SEEDS:
        a, b = (data["metrics"][s]["attribution"][m] for m in MODELS)
        cells.append(
            [
                s,
                f"{a['zero_mass_count']} / {b['zero_mass_count']}",
                percent(a["mean_foreground_attribution_share"]),
                percent(b["mean_foreground_attribution_share"]),
            ]
        )
    doc.table(
        "Tabel 9",
        "FG share rata-rata hanya pada peta nonzero; B=baseline, T=treatment.",
        cells,
        [25, 45, 50, 50],
        [s3(f"seed_{s}/metrics.json") for s in SEEDS],
        "10",
    )
    doc.p(
        "ReLU dapat menekan seluruh peta meski gradien tidak nol; share untuk massa nol "
        "tidak terdefinisi. FG share turun pada seed 42 dan naik pada dua seed lain. "
        "Kohort nonzero berbeda, sehingga rerata ini bukan perbandingan kausal berpasangan. "
        "Grad-CAM memberi bukti lokalisasi deskriptif, bukan bukti tunggal independensi latar."
    )
    if data["screenshot"]:
        doc.newpage()
        doc.p("10.1 Dashboard dan Artefak Visual Asli", "h1")
        doc.p(
            "Lima tab Streamlit menyediakan Overview, Dataset Explorer, Model Comparison, "
            "Evaluation, serta Explainability &amp; Export. Pemilihan seed dan pair mempertahankan "
            "identitas gambar; hasil dibaca dari artefak terverifikasi. Bila sumber belum tersedia, "
            "aplikasi menampilkan status kosong tanpa metrik buatan."
        )
        doc.figure(
            "Gambar 8",
            "Screenshot Chromium aplikasi Streamlit aktual, seed 123, "
            "tab Model Comparison. Sumber: workflow Stage 4 39140400975, 1440-model.png. "
            "Lima tab dan unduhan CSV/JSON/PNG diuji dalam bukti review asli.",
            data["screenshot"],
            [
                doc.source("stage4-dashboard-review", "dashboard-review/1440-model.png"),
                doc.source("stage4-dashboard-review", "dashboard-review/browser-review.json"),
            ],
            "10",
            max_height=112,
            seed=123,
            image_id="s123-image-002324",
            evidence_type="actual Chromium screenshot",
        )
        doc.figure(
            "Gambar 9",
            "Grafik asli cross_seed_summary.png dari Stage 3; mean dan sample SD "
            "untuk seed 42, 123, 2026. Gambar digunakan tanpa mengubah piksel atau hasilnya. "
            "Angka rinci dapat dibaca pada Tabel 5-6.",
            data["root"] / "figures/cross_seed_summary.png",
            [s3("figures/cross_seed_summary.png"), s3("summary.csv"), s3("summary.json")],
            "10.1",
            max_height=65,
            seeds=list(SEEDS),
            evidence_type="original scientific figure",
        )
    else:
        doc.p(
            "Screenshot tidak disertakan karena sumber Stage 4 opsional tidak diberikan. "
            "Tidak ada mockup pengganti. Dashboard tersedia melalui instruksi repository."
        )
    # Page 11.
    doc.newpage()
    doc.p("11. Diskusi dan Keterbatasan", "h1")
    doc.p(
        "Desain berpasangan mempertahankan foreground, inisialisasi, urutan training, "
        "arsitektur, optimizer, dan validation set. Perbedaan yang disengaja ialah assignment "
        "latar training. Dengan kontrol tersebut, perbedaan performa konsisten dengan efek "
        "intervensi distribusi dalam sistem ini. Seleksi checkpoint memakai aturan identik, "
        "tetapi epoch terpilih dapat berbeda sebagai bagian prosedur yang telah ditentukan."
    )
    doc.p(
        "Efek tersebut tidak membuktikan bahwa model memahami konsep sapi seperti manusia "
        "atau mengabaikan semua informasi latar. Kepekaan probabilitas masih ada; satu "
        "kesalahan baru muncul; lokalisasi Grad-CAM tidak seragam bergeser menuju foreground. "
        "Interpretasi yang dapat dipertahankan adalah perbaikan ketahanan terukur pada "
        "perubahan latar yang disediakan generator."
    )
    limitations = [
        (
            "Validitas eksternal",
            "Citra sintetis prosedural, hanya renderer sapi/kuda dan latar rumput/pantai. Generalisasi terhadap foto nyata belum diuji.",
        ),
        (
            "Replikasi terbatas",
            "Hanya tiga seed yang bersama-sama mengubah data dan inisialisasi; variasi masing-masing faktor tidak dipisahkan.",
        ),
        (
            "Dependensi observasi",
            "Dua citra test berbagi foreground. Tidak ada analisis signifikansi konfirmatori atau klaim interval kepercayaan.",
        ),
        (
            "Artefak generator",
            "Pola rendering, resolusi rendah, compositing, dan tepi alfa dapat menyediakan petunjuk lain. Audit kebocoran tidak meniadakan seluruh bias generator.",
        ),
        (
            "Bukti atribusi",
            "Peta Grad-CAM kasar, normalisasi per citra, zero maps, dan kohort nonzero berbeda membatasi interpretasi kausal.",
        ),
        (
            "Kesalahan residual",
            f"Treatment masih memiliki {residual} prediksi salah dan {new} kesalahan baru dibanding baseline; tidak ada klaim robustness universal.",
        ),
    ]
    doc.table(
        "Tabel 10",
        "Batas interpretasi yang berlaku untuk seluruh hasil.",
        [["Aspek", "Keterbatasan"], *limitations],
        [36, 134],
        [s3("summary.json"), repo("docs/methodology.md")],
        "11",
    )
    doc.p(
        "Pengembangan berikutnya dapat menguji foto nyata dengan pemisahan sumber ketat, "
        "spesies dan latar tambahan, lebih banyak seed, serta variasi data dan inisialisasi "
        "secara terpisah. Usulan ini belum dijalankan dan tidak termasuk bukti hasil saat ini."
    )
    doc.p("12. Kesimpulan", "h1")
    doc.p(
        "CowBias Lab menjawab tugas melalui sistem yang benar-benar membangkitkan data, "
        "melatih dua CNN, menerapkan background randomization seimbang, dan memvisualisasikan "
        "perilaku hasilnya. Treatment meningkatkan overall accuracy dan WGA serta menurunkan "
        "flip rate pada ketiga seed dibanding baseline berkorelasi kuat. Prediksi asli, pasangan "
        "counterfactual, dan Grad-CAM dapat diperiksa melalui dashboard dan laporan. "
        "Kesimpulan ini berlaku pada eksperimen sintetis terkontrol; kemampuan mengenali "
        "foto sapi sungguhan di pantai belum dibuktikan."
    )
    # Page 12.
    doc.newpage()
    doc.p("13. Referensi dan Bukti Reproduksibilitas", "h1")
    references = [
        (
            "[1] Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., &amp; Wichmann, F. A. (2020). Shortcut learning in deep neural networks. Nature Machine Intelligence, 2, 665-673.",
            "https://doi.org/10.1038/s42256-020-00257-z",
        ),
        (
            "[2] Xiao, K., Engstrom, L., Ilyas, A., &amp; Madry, A. (2021). Noise or Signal: The Role of Image Backgrounds in Object Recognition. ICLR. Preprint awal 2020, arXiv:2006.09994.",
            "https://arxiv.org/abs/2006.09994",
        ),
        (
            "[3] Sagawa, S., Koh, P. W., Hashimoto, T. B., &amp; Liang, P. (2020). Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization. ICLR. Preprint awal 2019, arXiv:1911.08731.",
            "https://arxiv.org/abs/1911.08731",
        ),
        (
            "[4] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &amp; Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. ICCV, 618-626.",
            "https://doi.org/10.1109/ICCV.2017.74",
        ),
    ]
    for ref, url in references:
        doc.p(ref + f' <link href="{url}" color="#176b66">{url}</link>', "caption")
    doc.p("13.1 Provenance dan akses bukti", "h2")
    for name, spec in ARCHIVES.items():
        url = f"https://github.com/{REPOSITORY}/actions/runs/{spec['workflow_run_id']}"
        doc.p(
            f'<b>{name}</b>: workflow <link href="{url}" color="#176b66">'
            f"{spec['workflow_run_id']}</link>; artifact {spec['artifact_id']}.<br/>"
            f"Source commit: {spec['source_commit']}<br/>SHA-256: {spec['sha256']}<br/>"
            f"Kedaluwarsa Actions: {spec['expires_at']}.",
            "caption",
        )
    doc.p(
        f"Run evaluasi: <b>{RUN_ID}</b>. Stage 2 dijalankan pada CPU, Python "
        "3.12.15, PyTorch 2.14.1+cpu, NumPy 2.5.4, Pillow 12.3.0, PyYAML 6.0.3; "
        "Stage 3 menambahkan Matplotlib 3.10.8. Metadata mencatat dua thread dan algoritma "
        "deterministik. Kesamaan bit lintas versi/perangkat tidak dijanjikan.",
        "caption",
    )
    doc.p(
        "<b>Preservasi jangka panjang: PENDING.</b> Repository bersifat publik; belum ada "
        "tujuan arsip durabel dengan kontrol akses yang diotorisasi. ZIP asli dan manifest "
        "hash terverifikasi tersedia, tetapi workspace sementara dan URL Actions bukan arsip "
        "permanen. Arsip harus disalin ke tujuan yang disetujui sebelum tanggal retensi berakhir.",
        "caption",
    )
    doc.p("13.2 Reproduksi laporan dan pemeriksaan", "h2")
    doc.p(
        "Jalankan scripts/archive_evidence.py untuk memulihkan ZIP asli; "
        "scripts/export_report.py dengan --run-id eksplisit, --artifact-root, --training-run, "
        "--archive-dir, dan --output. Screenshot opsional memakai --stage4-root. Perintah "
        "lengkap, fresh-clone smoke, dan verifier tersedia pada docs/reproducibility.md.",
        "caption",
    )
    doc.p(
        "docs/evidence-manifest.json memetakan tabel/gambar ke path internal, seed, "
        "image/pair ID, checksum, dan halaman. Generator menolak file tidak kompatibel. "
        "Riwayat training dan angka full run tidak diganti oleh smoke test Stage 5. "
        "Laporan persiapan/prototipe sebelumnya tidak dipakai sebagai sumber hasil.",
        "caption",
    )
    doc.p(
        "Belum ada LICENSE repository yang memberikan lisensi penggunaan umum. Kode dan "
        "artefak proyek perlu atribusi; lisensi dependensi tetap milik masing-masing. "
        "Kontribusi mahasiswa/pengembang serta penggunaan bantuan AI harus dinyatakan "
        "secara akurat sesuai kebijakan akademik sebelum pengumpulan.",
        "caption",
    )
    doc.build()
    return doc.entries


def generate(*, output, manifest, **inputs):
    output, manifest = Path(output), Path(manifest)
    if output.resolve() == manifest.resolve() or output.exists() or manifest.exists():
        raise ValueError("Report output path collision; choose NEW PDF and manifest paths")
    # Never write generated files within any source evidence directory.
    for key in ("artifact_root", "training_run", "archive_dir", "stage4_root"):
        if inputs.get(key):
            source = Path(inputs[key]).resolve()
            if output.resolve().is_relative_to(source) or manifest.resolve().is_relative_to(source):
                raise ValueError("Report outputs must be outside original evidence directories")
    data = load_evidence(**inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cowbias-report-") as tmp:
        folder = Path(tmp)
        pdf = folder / "report.pdf"
        entries = compose(data, pdf, folder)
        result = {
            "schema_version": 1,
            "expected_pages": 13 if data["screenshot"] else 12,
            "run_id": RUN_ID,
            "report_title": TITLE,
            "pdf_sha256": sha256(pdf),
            "generation_command": COMMAND,
            "report_sources": {
                str(p.relative_to(Path.cwd())): sha256(p)
                for p in [
                    Path.cwd() / "src/cowbias/report.py",
                    Path.cwd() / "src/cowbias/report_data.py",
                    Path.cwd() / "src/cowbias/evidence.py",
                ]
            },
            "environment": {
                "python": platform.python_version(),
                "packages": {p: version(p) for p in ("reportlab", "matplotlib", "numpy", "Pillow")},
            },
            "preservation": PRESERVATION,
            "archives": data["archives"]["archives"],
            "scientific_source_hashes": data["source_hashes"],
            "verification": data["verification"],
            "entries": entries,
            "scientific_totals": {
                "treatment_corrected": sum(
                    data["metrics"][s]["comparison"]["treatment_corrected"] for s in SEEDS
                ),
                "new_errors": sum(data["metrics"][s]["comparison"]["new_errors"] for s in SEEDS),
            },
        }
        from cowbias.report_verification import verify_pdf

        result["pdf_checks"] = verify_pdf(pdf, result)
        with output.open("xb") as handle:
            handle.write(pdf.read_bytes())
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return result["pdf_checks"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        "--run",
        dest="run_id",
        required=True,
        help="Exact verified evaluation run ID; never newest-run discovery",
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/evidence/stage3-scientific-results")
    )
    parser.add_argument(
        "--training-run", type=Path, default=Path("artifacts/evidence/stage2-trained-models")
    )
    parser.add_argument("--archive-dir", type=Path, default=Path("artifacts/evidence"))
    parser.add_argument(
        "--stage4-root", type=Path, help="Optional original extracted Stage 4 screenshot evidence"
    )
    parser.add_argument("--output", type=Path, default=Path("docs/task-3-report.pdf"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/evidence-manifest.json"))
    args = parser.parse_args(argv)
    try:
        result = generate(**vars(args))
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Report generation failed: {exc}")
        return 1
