"""Real Chromium acceptance: identity, viewports, downloads and screenshot evidence."""

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from cowbias.dashboard.data import image_array, load_arrays, load_run, load_seed

HEADINGS = {
    "Overview": "Can a CNN recognize the animal when the background changes?",
    "Dataset Explorer": "Inspect the controlled test set",
    "Model Comparison": "Change the background. Keep the animal.",
    "Evaluation": "Held-out evidence, including errors",
    "Explainability & Export": "Genuine Grad-CAM, with its limits",
}


@contextmanager
def server(root, port, output, fixture=False):
    env = dict(os.environ, COWBIAS_EVAL_ROOT=str(root), COWBIAS_FIXTURE="1" if fixture else "0")
    with (output / f"server-{port}.log").open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app/streamlit_app.py",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(port),
                "--server.fileWatcherType",
                "none",
                "--browser.gatherUsageStats",
                "false",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(150):
                if proc.poll() is not None:
                    raise RuntimeError("Streamlit exited; see server log")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/_stcore/health", timeout=1
                    ):
                        break
                except OSError:
                    time.sleep(0.2)  # Bounded HTTP readiness polling, not widget synchronization.
            else:
                raise RuntimeError("Streamlit HTTP readiness timed out")
            yield f"http://127.0.0.1:{port}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def settle(page):
    from playwright.sync_api import expect

    expect(page.get_by_test_id("stApp")).to_have_attribute(
        "data-test-script-state", "notRunning", timeout=30000
    )
    assert page.get_by_test_id("stException").count() == 0
    assert "Scientific artifact unavailable or invalid" not in page.locator("body").inner_text()


def select(page, label, value):
    from playwright.sync_api import expect

    combo = page.get_by_role("combobox", name=label, exact=True)
    if combo.input_value() == value:
        return
    combo.click()
    combo.fill("")
    combo.fill(value)
    option = page.get_by_role("option", name=value, exact=True)
    expect(option).to_be_visible()
    option.click()
    expect(combo).to_have_value(value)


def tab(page, name):
    from playwright.sync_api import expect

    page.get_by_role("tab", name=name, exact=True).click()
    expect(page.get_by_role("heading", name=HEADINGS[name], exact=True)).to_be_visible(
        timeout=30000
    )
    settle(page)


def selection(page, seed, *, image=None, outcome=None, bg=None, pair=None):
    from playwright.sync_api import expect

    marker = page.get_by_test_id("stCaptionContainer").filter(has_text=f"Selection · seed={seed} ·")
    for key, value in (("image", image), ("outcome", outcome), ("background", bg), ("pair", pair)):
        if value is not None:
            expect(marker).to_contain_text(f"{key}={value}", timeout=30000)
    settle(page)
    return marker.inner_text()


def capture(page, section, path):
    if section in ("Dataset Explorer", "Model Comparison", "Explainability & Export"):
        page.get_by_test_id("stImage").first.scroll_into_view_if_needed()
    elif section == "Evaluation":
        page.get_by_role(
            "heading", name="Accuracy in all four groups", exact=True
        ).scroll_into_view_if_needed()
    else:
        page.get_by_role("heading", name="CowBias Lab", exact=True).scroll_into_view_if_needed()
    page.screenshot(path=str(path), full_page=True)


def save_download(page, label, target):
    with page.expect_download() as event:
        page.get_by_role("button", name=label, exact=True).click()
    download = event.value
    assert download.failure() is None
    download.save_as(str(target))
    assert target.is_file() and target.stat().st_size > 0
    return target.read_bytes()


def review(root, output, fixture=False):
    from playwright.sync_api import expect, sync_playwright

    run = load_run(root, allow_fixture=fixture)
    output.mkdir(parents=True, exist_ok=True)
    downloads = output / "downloads"
    downloads.mkdir(exist_ok=True)
    report = {
        "run_id": run.config["run_id"],
        "context": run.receipt["context"],
        "checks": [],
        "viewports": [],
        "exports": [],
        "page_errors": [],
    }
    with server(root, 8511, output, fixture) as url, sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
            page.on("pageerror", lambda e: report["page_errors"].append(str(e)))
            start = time.monotonic()
            page.goto(url)
            expect(
                page.get_by_role("heading", name=HEADINGS["Overview"], exact=True)
            ).to_be_visible(timeout=30000)
            settle(page)
            report["first_meaningful_screen_seconds"] = time.monotonic() - start
            for name in HEADINGS:
                tab(page, name)
                capture(page, name, output / f"1440-{name.lower().split()[0]}.png")
            report["checks"].append("all five tabs on verified full run")
            tab(page, "Model Comparison")
            # Open native expander to select case filters.
            page.get_by_text("Filter observed cases", exact=True).click()
            from cowbias.dashboard.ui import OUTCOMES

            for seed in run.seeds:
                select(page, "Dataset seed", str(seed))
                expect(
                    page.get_by_text(re.compile(f"Active seed {seed} · Verification PASS"))
                ).to_be_visible()
                settle(page)
                data = load_seed(run, seed)
                for outcome in (
                    "treatment_only",
                    "baseline_only",
                    "both_correct",
                    "both_incorrect",
                ):
                    select(page, "Observed outcome", OUTCOMES[outcome])
                    options = data.pair_options(outcome=outcome)
                    if not options:
                        expect(
                            page.get_by_text(
                                f"Empty selection · seed={seed} · outcome={outcome}", exact=True
                            )
                        ).to_be_visible()
                        settle(page)
                        report["checks"].append(f"seed {seed}/{outcome}: honest empty category")
                        continue
                    selection(page, seed, outcome=outcome)
                    pair = options[0]
                    select(page, "Foreground pair", pair)
                    selection(page, seed, pair=pair, outcome=outcome)
                    for bg in ("grass", "beach"):
                        page.get_by_test_id("stRadio").get_by_text(bg.title(), exact=True).click()
                        image = data.image_id(pair, bg)
                        selection(page, seed, image=image, pair=pair, bg=bg)
                        assert (
                            page.get_by_role(
                                "combobox", name="Foreground pair", exact=True
                            ).input_value()
                            == pair
                        )
                        expected = [
                            f"{data.prediction(image, m)['prob_cow']:.4%}"
                            for m in ("baseline", "treatment")
                        ]
                        expect(page.get_by_test_id("stMetricValue")).to_have_text(expected)
                    if outcome == "baseline_only":
                        capture(
                            page,
                            "Model Comparison",
                            output / f"seed-{seed}-new-treatment-error.png",
                        )
                    report["checks"].append(
                        f"seed {seed}/{outcome}: foreground stable; both exact predictions"
                    )
                print(f"Browser seed {seed}: PASS", flush=True)
            select(page, "Dataset seed", str(run.seeds[0]))
            expect(
                page.get_by_text(re.compile(f"Active seed {run.seeds[0]} · Verification PASS"))
            ).to_be_visible()
            settle(page)
            tab(page, "Explainability & Export")
            select(page, "Attribution case", "First baseline zero map")
            data = load_seed(run, run.seeds[0])
            zero_ids = [
                i for (m, i), a in data.attribution.items() if m == "baseline" and a["zero_mass"]
            ]
            if zero_ids:
                image = zero_ids[0]
                selection(page, data.seed, image=image)
                expect(page.get_by_text("Zero map: Yes", exact=True)).to_be_visible()
                expect(page.get_by_text(re.compile("Foreground share: Undefined"))).to_be_visible()
                raw = save_download(
                    page, "Download baseline overlay PNG", downloads / "zero-baseline-overlay.png"
                )
                with Image.open(io.BytesIO(raw)) as img:
                    meta = json.loads(img.info["CowBias provenance"])
                    assert (
                        meta["image_id"] == image
                        and meta["seed"] == data.seed
                        and meta["model"] == "baseline"
                    )
                    assert np.array_equal(
                        np.array(img), image_array(load_arrays(data), image, "images")
                    )
                    img.verify()
                report["exports"].append(
                    "zero-map PNG equals original model input; full target/provenance metadata"
                )
            capture(page, "Explainability & Export", output / "zero-gradcam.png")
            for seed in run.seeds[:2]:
                select(page, "Dataset seed", str(seed))
                expect(
                    page.get_by_text(re.compile(f"Active seed {seed} · Verification PASS"))
                ).to_be_visible()
                settle(page)
                for suffix in ("predictions.csv", "metrics.json"):
                    name = f"seed_{seed}/{suffix}"
                    select(page, "Evidence file", name)
                    # Native download button is rendered with the selected verified file bytes.
                    settle(page)
                    raw = save_download(
                        page, "Download scientific file", downloads / f"seed-{seed}-{suffix}"
                    )
                    assert raw == run.export(name), name
                    if suffix.endswith(".csv"):
                        rows = list(csv.DictReader(io.StringIO(raw.decode())))
                        assert rows and all(
                            int(r["seed"]) == seed and r["run_id"] == run.config["run_id"]
                            for r in rows
                        )
                    else:
                        assert json.loads(raw)["seed"] == seed
                    report["exports"].append(name + ": downloaded, reopened, exact source bytes")
                figure = f"seed_{seed}/figures/accuracy_by_group.png"
                select(page, "Scientific figure", figure)
                settle(page)
                raw = save_download(
                    page, "Download scientific figure PNG", downloads / f"seed-{seed}-figure.png"
                )
                assert raw == run.export(figure)
                with Image.open(io.BytesIO(raw)) as img:
                    img.verify()
                report["exports"].append(figure + ": downloaded, reopened, exact source bytes")
            for width in (1440, 1280, 390):
                page.set_viewport_size({"width": width, "height": 1000})
                sidebar = page.get_by_test_id("stSidebar")
                close = page.get_by_test_id("stSidebarCollapseButton")
                if (
                    width == 390
                    and sidebar.get_attribute("data-collapsed") == "false"
                    and close.is_visible()
                ):
                    close.click()
                for name in HEADINGS:
                    tab(page, name)
                    assert not page.evaluate(
                        "document.documentElement.scrollWidth > innerWidth + 1"
                    )
                    assert page.get_by_test_id("stMain").evaluate(
                        "e=>e.scrollWidth <= e.clientWidth + 1"
                    )
                    capture(page, name, output / f"{width}-{name.lower().split()[0]}.png")
                report["viewports"].append(
                    {"width": width, "sections": 5, "horizontal_overflow": False, "status": "PASS"}
                )
                print(f"Browser viewport {width}: PASS", flush=True)
        except Exception:
            if "page" in locals():
                page.screenshot(path=str(output / "failure-state.png"), full_page=True)
            raise
        finally:
            browser.close()
    with (
        tempfile.TemporaryDirectory() as temp,
        server(Path(temp) / "missing", 8512, output) as url,
        sync_playwright() as p,
    ):
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        try:
            page = browser.new_page(viewport={"width": 390, "height": 844})
            page.goto(url)
            expect(page.get_by_text("Experiment not yet run", exact=True)).to_be_visible(
                timeout=30000
            )
            assert page.get_by_test_id("stException").count() == 0
            page.screenshot(path=str(output / "390-missing.png"), full_page=True)
        finally:
            browser.close()
    assert not report["page_errors"], report["page_errors"]
    report["checks"].append("missing-artifact browser state; no unhandled Streamlit exceptions")
    report["status"] = "PASS"
    (output / "browser-review.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dashboard-review"))
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(review(args.run, args.output, args.fixture), indent=2))
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "browser-failure.json").write_text(
            json.dumps({"status": "FAIL", "error": str(exc)}, indent=2)
        )
        raise


if __name__ == "__main__":
    main()
