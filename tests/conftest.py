from dataclasses import replace
from pathlib import Path

import pytest

from cowbias.config import load_config
from cowbias.data.pipeline import generate_dataset

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config():
    return load_config(REPO / "config/experiment.yaml")


@pytest.fixture(scope="session")
def small_config(config):
    # Real 97:3 per class with minorities, but fewer source files for integration tests.
    return replace(
        config,
        dataset=replace(config.dataset, train_size=200, val_size=8, test_size=8, image_size=32),
    )


@pytest.fixture(scope="session")
def generated(tmp_path_factory, small_config):
    root = tmp_path_factory.mktemp("integration") / "data"
    generate_dataset(small_config, root, 42)
    return root


@pytest.fixture(scope="session")
def audited(generated, small_config):
    pytest.importorskip("torch")
    from cowbias.training.data import AuditedData

    return AuditedData.open(generated, small_config, 42)


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory, small_config, generated):
    pytest.importorskip("torch")
    import shutil

    from cowbias.training.experiment import run_experiment

    parent = tmp_path_factory.mktemp("training")
    data_run = parent / "data-run"
    # Copy real Stage 1 assets; never mock generation, audit or optimization.
    shutil.copytree(generated, data_run / "seed_42" / "data")
    output = parent / "trained"
    run_experiment(
        small_config,
        output,
        seeds=[42],
        data_run=data_run,
        device="cpu",
    )
    return output
