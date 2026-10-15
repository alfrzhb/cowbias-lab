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
