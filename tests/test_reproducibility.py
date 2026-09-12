import numpy as np
import torch

from q_vla_forge.utils.reproducibility import DEFAULT_SEEDS, set_seed


def test_default_seeds() -> None:
    assert DEFAULT_SEEDS == (42, 123, 456)


def test_reproducibility() -> None:
    set_seed(42)

    numpy_first = np.random.rand(3)
    torch_first = torch.rand(3)

    set_seed(42)

    numpy_second = np.random.rand(3)
    torch_second = torch.rand(3)

    assert np.allclose(numpy_first, numpy_second)
    assert torch.allclose(torch_first, torch_second)
