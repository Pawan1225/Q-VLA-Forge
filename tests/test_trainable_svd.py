from __future__ import annotations

import torch
from torch import nn

from q_vla_forge.data import (
    Domain,
    SyntheticDrivingDataset,
)
from q_vla_forge.models import SharedVLAModel
from q_vla_forge.training import (
    TrainingConfig,
    train_supervised_instrumented,
)
from q_vla_forge.training.low_rank import (
    TrainableSVDLinear,
    convert_model_to_trainable_svd,
    trainable_parameter_count,
)
from q_vla_forge.utils.reproducibility import set_seed


def test_trainable_svd_factor_shapes() -> None:
    dense = nn.Linear(
        12,
        8,
        bias=True,
    )

    layer = TrainableSVDLinear.from_linear(
        dense,
        rank=3,
    )

    assert layer.u.shape == (
        8,
        3,
    )

    assert layer.singular_values.shape == (3,)

    assert layer.vh.shape == (
        3,
        12,
    )

    assert layer.bias is not None

    assert layer.bias.shape == (8,)


def test_forward_matches_truncated_reconstruction() -> None:
    set_seed(42)

    dense = nn.Linear(
        12,
        8,
        bias=True,
    )

    layer = TrainableSVDLinear.from_linear(
        dense,
        rank=3,
    )

    inputs = torch.randn(
        5,
        12,
    )

    reconstructed = layer.reconstructed_weight()

    expected = torch.nn.functional.linear(
        inputs,
        reconstructed,
        layer.bias,
    )

    actual = layer(inputs)

    assert torch.allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-5,
    )


def test_all_svd_factors_receive_gradients() -> None:
    set_seed(42)

    dense = nn.Linear(
        12,
        8,
        bias=True,
    )

    layer = TrainableSVDLinear.from_linear(
        dense,
        rank=3,
    )

    inputs = torch.randn(
        4,
        12,
    )

    loss = layer(inputs).square().mean()

    loss.backward()

    assert layer.u.grad is not None
    assert layer.singular_values.grad is not None
    assert layer.vh.grad is not None

    assert layer.bias is not None
    assert layer.bias.grad is not None


def test_factorization_reduces_layer_parameters() -> None:
    dense = nn.Linear(
        64,
        64,
        bias=True,
    )

    layer = TrainableSVDLinear.from_linear(
        dense,
        rank=8,
    )

    dense_count = sum(parameter.numel() for parameter in dense.parameters())

    structured_count = sum(parameter.numel() for parameter in layer.parameters())

    assert structured_count < dense_count


def test_model_conversion_does_not_mutate_original() -> None:
    set_seed(42)

    model = SharedVLAModel()

    before_types = {name: type(module) for name, module in model.named_modules()}

    structured, report = convert_model_to_trainable_svd(
        model,
        rank_fraction=0.50,
    )

    after_types = {name: type(module) for name, module in model.named_modules()}

    assert before_types == after_types
    assert structured is not model
    assert report.layers


def test_model_conversion_contains_trainable_svd() -> None:
    set_seed(42)

    model = SharedVLAModel()

    structured, report = convert_model_to_trainable_svd(
        model,
        rank_fraction=0.50,
    )

    modules = list(structured.modules())

    assert any(
        isinstance(
            module,
            TrainableSVDLinear,
        )
        for module in modules
    )

    assert len(report.layers) > 0


def test_whole_model_trainable_parameters_reduce() -> None:
    set_seed(42)

    model = SharedVLAModel()

    structured, report = convert_model_to_trainable_svd(
        model,
        rank_fraction=0.50,
    )

    assert report.structured_trainable_parameters < report.original_trainable_parameters

    assert (
        trainable_parameter_count(structured) == report.structured_trainable_parameters
    )


def test_only_profitable_layers_are_replaced() -> None:
    set_seed(42)

    model = SharedVLAModel()

    _, report = convert_model_to_trainable_svd(
        model,
        rank_fraction=0.50,
    )

    for layer in report.layers:
        assert layer.factor_parameters < layer.original_weight_parameters


def test_small_layers_remain_dense() -> None:
    model = nn.Sequential(
        nn.Linear(
            4,
            4,
        )
    )

    structured, report = convert_model_to_trainable_svd(
        model,
        rank_fraction=0.50,
        minimum_weight_parameters=1024,
    )

    assert isinstance(
        structured[0],
        nn.Linear,
    )

    assert not report.layers


def test_selected_layer_names_are_respected() -> None:
    model = nn.Sequential(
        nn.Linear(
            64,
            64,
        ),
        nn.ReLU(),
        nn.Linear(
            64,
            64,
        ),
    )

    structured, report = convert_model_to_trainable_svd(
        model,
        rank_fraction=0.50,
        minimum_weight_parameters=1,
        selected_layer_names={"0"},
    )

    assert isinstance(
        structured[0],
        TrainableSVDLinear,
    )

    assert isinstance(
        structured[2],
        nn.Linear,
    )

    assert [item.name for item in report.layers] == ["0"]


def test_structured_vla_forward_driving() -> None:
    set_seed(42)

    model = SharedVLAModel()

    structured, _ = convert_model_to_trainable_svd(
        model,
        rank_fraction=0.50,
    )

    samples = list(
        SyntheticDrivingDataset(
            size=4,
            seed=42,
        )
    )

    visual = torch.stack(
        [torch.from_numpy(sample.observation.visual) for sample in samples]
    ).float()

    state = torch.stack(
        [torch.from_numpy(sample.observation.state) for sample in samples]
    ).float()

    language = tuple(sample.observation.language_goal for sample in samples)

    output = structured(
        visual,
        state,
        language,
        Domain.AUTONOMOUS_DRIVING,
    )

    assert output.shape == (
        4,
        3,
    )

    assert torch.isfinite(output).all()


def test_trainable_svd_can_optimize() -> None:
    train_samples = list(
        SyntheticDrivingDataset(
            size=64,
            seed=42,
        )
    )

    validation_samples = list(
        SyntheticDrivingDataset(
            size=32,
            seed=1042,
        )
    )

    set_seed(42)

    model = SharedVLAModel()

    structured, _ = convert_model_to_trainable_svd(
        model,
        rank_fraction=0.50,
    )

    result = train_supervised_instrumented(
        model=structured,
        train_samples=train_samples,
        validation_samples=validation_samples,
        domain=Domain.AUTONOMOUS_DRIVING,
        config=TrainingConfig(
            epochs=2,
            batch_size=16,
            seed=42,
        ),
    )

    assert len(result.history) == 2

    assert all(torch.isfinite(torch.tensor(item.train_loss)) for item in result.history)


def test_bias_is_preserved_exactly() -> None:
    set_seed(42)

    dense = nn.Linear(
        12,
        8,
        bias=True,
    )

    assert dense.bias is not None

    original_bias = dense.bias.detach().clone()

    layer = TrainableSVDLinear.from_linear(
        dense,
        rank=3,
    )

    assert layer.bias is not None

    assert torch.equal(
        original_bias,
        layer.bias.detach(),
    )
