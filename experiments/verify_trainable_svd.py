from __future__ import annotations

from q_vla_forge.models import SharedVLAModel
from q_vla_forge.training import (
    TrainableSVDLinear,
    convert_model_to_trainable_svd,
    trainable_parameter_count,
)
from q_vla_forge.utils.reproducibility import set_seed


def main() -> None:
    set_seed(42)

    model = SharedVLAModel()

    baseline_parameters = trainable_parameter_count(model)

    structured, report = convert_model_to_trainable_svd(
        model,
        rank_fraction=0.50,
        minimum_weight_parameters=1024,
    )

    structured_parameters = trainable_parameter_count(structured)

    print()
    print("===== Q-VLA Forge " "Trainable SVD Verification =====")

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
            "  shape:",
            f"{layer.output_dim}x{layer.input_dim}",
        )

        print(
            "  rank:",
            layer.rank,
        )

        print(
            "  dense weight parameters:",
            layer.original_weight_parameters,
        )

        print(
            "  factor parameters:",
            layer.factor_parameters,
        )

        print(
            "  local parameter ratio:",
            f"{layer.local_parameter_ratio:.3f}x",
        )

        print(
            "  init reconstruction error:",
            f"{layer.relative_initialization_error:.6f}",
        )

    svd_modules = [
        name
        for name, module in structured.named_modules()
        if isinstance(
            module,
            TrainableSVDLinear,
        )
    ]

    if len(svd_modules) != len(report.layers):
        raise RuntimeError("SVD module/report count mismatch")

    if structured_parameters >= baseline_parameters:
        raise RuntimeError("trainable SVD did not " "reduce parameters")

    print()
    print("Trainable SVD modules:")

    for name in svd_modules:
        print(
            " ",
            name,
        )

    print()
    print("===== TRAINABLE SVD VERIFIED =====")


if __name__ == "__main__":
    main()
