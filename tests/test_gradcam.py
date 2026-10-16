import numpy as np
import pytest
import torch

from cowbias.evaluation.gradcam import attribution, final_activation, gradcam, target_score
from cowbias.models.cnn import SmallCNN
from cowbias.training.reproducibility import seed_everything, state_sha256


def test_real_gradient_signed_target_hook_cleanup_and_state():
    seed_everything(42)
    model = SmallCNN().eval()
    image = torch.randn(1, 3, 32, 32)
    copy = image.clone()
    before = state_sha256(model.state_dict())
    hooks = len(final_activation(model)._forward_hooks)
    heat, meta = gradcam(model, image, 1)
    assert heat.shape == (32, 32) and np.isfinite(heat).all()
    assert meta["gradient_l1"] > 0 and 0 <= heat.min() <= heat.max() <= 1
    assert len(final_activation(model)._forward_hooks) == hooks
    assert before == meta["state_before"] == meta["state_after"] == state_sha256(model.state_dict())
    assert torch.equal(image, copy) and all(p.grad is None for p in model.parameters())
    scores = torch.tensor([2.0], requires_grad=True)
    assert torch.autograd.grad(target_score(scores, 0), scores)[0].item() == -1
    assert target_score(scores, 1).item() == 2
    h0, m0 = gradcam(model, image, 0)
    assert np.isfinite(h0).all() and m0["score"] == "-logit"


def test_zero_heatmap_and_zero_gradient():
    model = SmallCNN()
    for p in model.parameters():
        p.data.zero_()
    heat, meta = gradcam(model, torch.zeros(1, 3, 32, 32), 0)
    assert np.count_nonzero(heat) == 0 and meta["zero_mass"]
    assert meta["gradient_l1"] == 0


@pytest.mark.parametrize("target", [-1, 2, "cow", True])
def test_invalid_gradcam_target(target):
    with pytest.raises(ValueError):
        gradcam(SmallCNN(), torch.zeros(1, 3, 32, 32), target)


def test_invalid_gradcam_image_and_hook_cleanup():
    model = SmallCNN()
    with pytest.raises(ValueError):
        gradcam(model, torch.zeros(2, 3, 32, 32), 1)
    with pytest.raises(ValueError):
        gradcam(model, torch.full((1, 3, 32, 32), float("nan")), 1)
    assert len(final_activation(model)._forward_hooks) == 0


def test_attribution_known_soft_mask_and_uniform_heat():
    mask = np.array([[0, 0.5], [0.5, 1]])
    result = attribution(np.ones((2, 2)), mask)
    assert result["foreground_attribution_share"] == 0.5
    assert result["background_attribution_share"] == 0.5
    assert result["foreground_area_fraction"] == 0.5
    assert result["foreground_attribution_enrichment"] == 1
    assert attribution(np.zeros((2, 2)), mask)["foreground_attribution_share"] is None
    assert (
        attribution(np.ones((2, 2)), np.zeros((2, 2)))["foreground_attribution_enrichment"] is None
    )


@pytest.mark.parametrize(
    "heat,mask",
    [
        (np.zeros((2, 2)), np.zeros((3, 3))),
        (np.full((2, 2), np.nan), np.zeros((2, 2))),
        (np.ones((2, 2)), np.full((2, 2), 1.1)),
        (np.full((2, 2), -1), np.zeros((2, 2))),
    ],
)
def test_invalid_attribution(heat, mask):
    with pytest.raises(ValueError):
        attribution(heat, mask)
