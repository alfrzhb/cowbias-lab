import pytest

torch = pytest.importorskip("torch")

from cowbias.models.cnn import SmallCNN, model_from_metadata
from cowbias.training.reproducibility import clone_state, seed_everything, state_sha256


def test_forward_backward_finite():
    seed_everything(42)
    model = SmallCNN()
    logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4,)
    loss = torch.nn.BCEWithLogitsLoss()(logits, torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.architecture() == model_from_metadata(model.architecture()).architecture()


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros(1, 1, 32, 32),
        torch.zeros(0, 3, 32, 32),
        torch.full((1, 3, 32, 32), float("nan")),
        torch.zeros(1, 3, 32, 32, dtype=torch.float64),
    ],
)
def test_invalid_inputs(images):
    with pytest.raises(ValueError):
        SmallCNN()(images)


def test_seeded_initialization():
    seed_everything(42)
    a = clone_state(SmallCNN())
    seed_everything(42)
    b = clone_state(SmallCNN())
    assert all(torch.equal(a[k], b[k]) for k in a)
    seed_everything(123)
    assert state_sha256(a) != state_sha256(clone_state(SmallCNN()))
