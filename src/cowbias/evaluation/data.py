"""Explicit, audited test-only adapter. Stage 2 train/val protection is unchanged."""

import numpy as np
import torch
from torch.utils.data import Dataset

from cowbias.data.dataset import ManifestDataset
from cowbias.training.data import AuditedData


class TestDataset(Dataset):
    __test__ = False

    def __init__(self, audited: AuditedData, *, split="test", variant="shared_eval"):
        if not isinstance(audited, AuditedData) or audited.report.get("status") != "PASS":
            raise ValueError("A successful current Stage 1 audit is required")
        if split != "test" or variant != "shared_eval":
            raise ValueError("Evaluation adapter only permits test/shared_eval")
        audited.assert_unchanged()
        self.audited = audited
        self.source = ManifestDataset(
            audited.root,
            split=split,
            variant=variant,
            image_size=audited.config.dataset.model_image_size,
        )
        self.records = self.source.records
        samples = [self.source[i] for i in range(len(self.source))]
        # Exactly Stage 2 preprocessing; ManifestDataset uses Pillow bilinear RGB resize.
        self.images = torch.stack(
            [
                torch.from_numpy(s["image"].astype(np.float32))
                .permute(2, 0, 1)
                .contiguous()
                .div(255)
                .sub(0.5)
                .div(0.5)
                for s in samples
            ]
        )
        # Nearest resampling preserves alpha values, including existing soft-alpha pixels.
        self.masks = np.stack([s["mask"].astype(np.float32) / 255 for s in samples])
        if len(self.records) != audited.config.dataset.test_size:
            raise ValueError("Unexpected held-out test count")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.images[index]
