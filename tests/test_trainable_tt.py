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
from q_vla_forge.training.tensor_network import (
    TrainableTTLinear,
    convert_model_to_trainable_tt,
    trainable_parameter_count,
)
from q_vla_forge.utils.reproducibility import set_seed


def test_tt_layer_has_three_cores() -> None:
    set_seed(42)

    dense = nn.Linear(
        64,
        64,
    )

    layer = TrainableTTLinear.from_linear(
        dense,
        max_rank=4,
        tensor_order=3,
    )

    assert len(layer.cores) == 3


def test_tt_open_boundary_ranks() -> None:
    set_seed(42)

    dense = nn.Linear(
        64,
        64,
    )

    layer = TrainableTTLinear.from_linear(
        dense,
        max_rank=4,
        tensor_order=3,
    )

    assert layer.ranks[0] == 1
    assert layer.ranks[-1] == 1


def test_tt_forward_matches_reconstructed_weight() -> None:
    set_seed(42)

    dense = nn.Linear(
        64,
        64,
        bias=True,
    )

    layer = TrainableTTLinear.from_linear(
        dense,
        max_rank=4,
        tensor_order=3,
    )

    inputs = torch.randn(
        5,
        64,
    )

    weight = layer.reconstructed_weight()

    expected = torch.nn.functional.linear(
        inputs,
        weight,
        layer.bias,
    )

    actual = layer(inputs)

    assert torch.allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-5,
    )


def test_all_tt_cores_receive_gradients() -> None:
    set_seed(42)

    dense = nn.Linear(
        64,
        64,
    )

    layer = TrainableTTLinear.from_linear(
        dense,
        max_rank=4,
        tensor_order=3,
    )

    inputs = torch.randn(
        4,
        64,
    )

    loss = layer(inputs).square().mean()

    loss.backward()

    assert all(core.grad is not None for core in layer.cores)


def test_tt_bias_receives_gradient() -> None:
    set_seed(42)

    dense = nn.Linear(
        64,
        64,
        bias=True,
    )

    layer = TrainableTTLinear.from_linear(
        dense,
        max_rank=4,
    )

    layer(
        torch.randn(
            4,
            64,
        )
    ).square().mean().backward()

    assert layer.bias is not None
    assert layer.bias.grad is not None


def test_tt_has_no_dense_weight_parameter() -> None:
    set_seed(42)

    dense = nn.Linear(
        64,
        64,
    )

    layer = TrainableTTLinear.from_linear(
        dense,
        max_rank=4,
    )

    names = {name for name, _ in layer.named_parameters()}

    assert "weight" not in names

    assert any(name.startswith("cores.") for name in names)


def test_tt_core_count_can_reduce_parameters() -> None:
    set_seed(42)

    dense = nn.Linear(
        64,
        64,
    )

    layer = TrainableTTLinear.from_linear(
        dense,
        max_rank=4,
    )

    assert layer.core_parameter_count < dense.weight.numel()


def test_conversion_does_not_mutate_original() -> None:
    set_seed(42)

    model = SharedVLAModel()

    original_types = {name: type(module) for name, module in model.named_modules()}

    structured, report = convert_model_to_trainable_tt(
        model,
        max_rank=4,
    )

    current_types = {name: type(module) for name, module in model.named_modules()}

    assert original_types == current_types
    assert structured is not model
    assert report.layers


def test_converted_model_contains_tt_layers() -> None:
    set_seed(42)

    model = SharedVLAModel()

    structured, report = convert_model_to_trainable_tt(
        model,
        max_rank=4,
    )

    assert any(
        isinstance(
            module,
            TrainableTTLinear,
        )
        for module in structured.modules()
    )

    assert len(report.layers) > 0


def test_whole_model_parameter_count_reduces() -> None:
    set_seed(42)

    model = SharedVLAModel()

    structured, report = convert_model_to_trainable_tt(
        model,
        max_rank=4,
    )

    assert report.structured_trainable_parameters < report.original_trainable_parameters

    assert (
        trainable_parameter_count(structured) == report.structured_trainable_parameters
    )


def test_only_profitable_tt_layers_are_replaced() -> None:
    set_seed(42)

    model = SharedVLAModel()

    _, report = convert_model_to_trainable_tt(
        model,
        max_rank=4,
    )

    for layer in report.layers:
        assert layer.core_parameters < layer.original_weight_parameters


def test_selected_layers_are_respected() -> None:
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

    structured, report = convert_model_to_trainable_tt(
        model,
        max_rank=4,
        minimum_weight_parameters=1,
        selected_layer_names={
            "0",
        },
    )

    assert isinstance(
        structured[0],
        TrainableTTLinear,
    )

    assert isinstance(
        structured[2],
        nn.Linear,
    )

    assert [layer.name for layer in report.layers] == [
        "0",
    ]


def test_small_layer_remains_dense() -> None:
    model = nn.Sequential(
        nn.Linear(
            4,
            4,
        )
    )

    structured, report = convert_model_to_trainable_tt(
        model,
        max_rank=2,
        minimum_weight_parameters=1024,
    )

    assert isinstance(
        structured[0],
        nn.Linear,
    )

    assert not report.layers


def test_structured_vla_forward() -> None:
    set_seed(42)

    model = SharedVLAModel()

    structured, _ = convert_model_to_trainable_tt(
        model,
        max_rank=4,
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


def test_tt_model_can_train() -> None:
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

    dense_model = SharedVLAModel()

    structured, _ = convert_model_to_trainable_tt(
        dense_model,
        max_rank=4,
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
        64,
        64,
        bias=True,
    )

    assert dense.bias is not None

    original_bias = dense.bias.detach().clone()

    layer = TrainableTTLinear.from_linear(
        dense,
        max_rank=4,
    )

    assert layer.bias is not None

    assert torch.equal(
        original_bias,
        layer.bias.detach(),
    )
