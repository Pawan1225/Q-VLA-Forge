from __future__ import annotations

from q_vla_forge.models import SharedVLAModel
from q_vla_forge.training import (
    TrainableTTLinear,
    convert_model_to_trainable_tt,
)
from q_vla_forge.utils.reproducibility import set_seed


def _count_trainable(
    model,
) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def main() -> None:
    set_seed(42)

    dense_model = SharedVLAModel()

    baseline_parameters = _count_trainable(dense_model)

    structured_model, report = convert_model_to_trainable_tt(
        dense_model,
        max_rank=4,
        minimum_weight_parameters=1024,
        tensor_order=3,
    )

    structured_parameters = _count_trainable(structured_model)

    print()
    print("===== Q-VLA Forge " "Trainable TT/MPS Verification =====")

    print()
    print(
        "Method:",
        "trainable_tt_mps",
    )

    print(
        "Family:",
        "quantum-inspired",
    )

    print(
        "Representation:",
        "TT / open-boundary MPS",
    )

    print(
        "Quantum hardware:",
        report.quantum_hardware_used,
    )

    print()
    print(
        "Baseline trainable parameters:",
        baseline_parameters,
    )

    print(
        "Structured trainable parameters:",
        structured_parameters,
    )

    print(
        "Whole-model parameter ratio:",
        f"{report.parameter_ratio:.3f}x",
    )

    print(
        "Whole-model reduction:",
        f"{report.parameter_reduction_percent:.3f}%",
    )

    print()
    print(
        "Requested rank:",
        report.requested_rank,
    )

    print(
        "Tensor order:",
        report.tensor_order,
    )

    print(
        "Replaced layers:",
        len(report.layers),
    )

    for layer in report.layers:
        print()
        print(
            "Layer:",
            layer.name,
        )

        print(
            "  matrix shape:",
            f"{layer.output_dim}x{layer.input_dim}",
        )

        print(
            "  row factors:",
            layer.row_factors,
        )

        print(
            "  column factors:",
            layer.column_factors,
        )

        print(
            "  tensor shape:",
            layer.tensor_shape,
        )

        print(
            "  requested rank:",
            layer.requested_rank,
        )

        print(
            "  actual ranks:",
            layer.actual_ranks,
        )

        print(
            "  dense weight parameters:",
            layer.original_weight_parameters,
        )

        print(
            "  TT core parameters:",
            layer.core_parameters,
        )

        print(
            "  local parameter ratio:",
            f"{layer.local_parameter_ratio:.3f}x",
        )

        print(
            "  initialization error:",
            f"{layer.relative_initialization_error:.6f}",
        )

    tt_modules = [
        name
        for name, module in structured_model.named_modules()
        if isinstance(
            module,
            TrainableTTLinear,
        )
    ]

    if len(tt_modules) != len(report.layers):
        raise RuntimeError("TT module/report count mismatch")

    if structured_parameters >= baseline_parameters:
        raise RuntimeError("trainable TT did not reduce model parameters")

    for (
        name,
        module,
    ) in structured_model.named_modules():
        if not isinstance(
            module,
            TrainableTTLinear,
        ):
            continue

        parameter_names = {
            parameter_name for parameter_name, _ in module.named_parameters()
        }

        if "weight" in parameter_names:
            raise RuntimeError(f"{name} contains a dense " "trainable weight")

    print()
    print("Trainable TT/MPS modules:")

    for name in tt_modules:
        print(
            " ",
            name,
        )

    print()
    print("===== TRAINABLE TT/MPS VERIFIED =====")


if __name__ == "__main__":
    main()
