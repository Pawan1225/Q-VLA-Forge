"""Compare seed-42 compression selections across driving and robotics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COMPRESSION_DIR = Path("results") / "compression"

DRIVING_PATH = COMPRESSION_DIR / "driving-compression-selection.json"

ROBOTICS_PATH = COMPRESSION_DIR / "robotics-compression-selection.json"

OUTPUT_PATH = COMPRESSION_DIR / "cross-domain-compression-selection.json"


def _load(
    path: Path,
) -> dict[str, Any]:
    """Load one selection artifact."""
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def _selected(
    payload: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Return one selected method candidate."""
    return payload[key]["selected"]


def _method_comparison(
    *,
    family: str,
    driving: dict[str, Any],
    robotics: dict[str, Any],
) -> dict[str, Any]:
    """Compare one compression family across both domains."""
    same_configuration = (
        driving["configuration_label"] == robotics["configuration_label"]
    )

    both_feasible = bool(driving["pilot_feasible"]) and bool(robotics["pilot_feasible"])

    return {
        "family": family,
        "same_configuration_selected": (same_configuration),
        "both_domains_pilot_feasible": (both_feasible),
        "driving": {
            "experiment_id": (driving["experiment_id"]),
            "configuration": (driving["configuration_label"]),
            "compression_ratio": (driving["compression_ratio"]),
            "mse_change_percent": (driving["mse_change_percent"]),
            "mae_change_percent": (driving["mae_change_percent"]),
            "pilot_feasible": (driving["pilot_feasible"]),
        },
        "robotics": {
            "experiment_id": (robotics["experiment_id"]),
            "configuration": (robotics["configuration_label"]),
            "compression_ratio": (robotics["compression_ratio"]),
            "mse_change_percent": (robotics["mse_change_percent"]),
            "mae_change_percent": (robotics["mae_change_percent"]),
            "pilot_feasible": (robotics["pilot_feasible"]),
        },
    }


def main() -> None:
    """Build cross-domain compression-selection evidence."""
    driving = _load(DRIVING_PATH)

    robotics = _load(ROBOTICS_PATH)

    if driving["seed"] != robotics["seed"]:
        raise ValueError(
            "driving and robotics selections " "must use the same exploratory seed"
        )

    if driving["minimum_compression_ratio"] != robotics["minimum_compression_ratio"]:
        raise ValueError("compression thresholds differ " "between domains")

    if (
        driving["maximum_mse_increase_percent"]
        != robotics["maximum_mse_increase_percent"]
    ):
        raise ValueError("MSE thresholds differ " "between domains")

    comparisons = [
        _method_comparison(
            family="int8",
            driving=_selected(
                driving,
                "int8",
            ),
            robotics=_selected(
                robotics,
                "int8",
            ),
        ),
        _method_comparison(
            family="svd",
            driving=_selected(
                driving,
                "svd",
            ),
            robotics=_selected(
                robotics,
                "svd",
            ),
        ),
        _method_comparison(
            family="tt_mps",
            driving=_selected(
                driving,
                "tensor_network",
            ),
            robotics=_selected(
                robotics,
                "tensor_network",
            ),
        ),
    ]

    shared_configuration_families = [
        comparison["family"]
        for comparison in comparisons
        if comparison["same_configuration_selected"]
    ]

    cross_domain_feasible_families = [
        comparison["family"]
        for comparison in comparisons
        if comparison["both_domains_pilot_feasible"]
    ]

    payload = {
        "seed": driving["seed"],
        "minimum_compression_ratio": (driving["minimum_compression_ratio"]),
        "maximum_mse_increase_percent": (driving["maximum_mse_increase_percent"]),
        "comparisons": comparisons,
        "shared_configuration_families": (shared_configuration_families),
        "cross_domain_feasible_families": (cross_domain_feasible_families),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("===== Cross-Domain " "Compression Selection =====")

    print()

    for comparison in comparisons:
        print(comparison["family"].upper())

        print(
            "  Driving:",
            comparison["driving"]["configuration"],
        )

        print(
            "  Robotics:",
            comparison["robotics"]["configuration"],
        )

        print(
            "  Same configuration:",
            comparison["same_configuration_selected"],
        )

        print(
            "  Both pilot feasible:",
            comparison["both_domains_pilot_feasible"],
        )

        print()

    print(
        "Shared configuration families:",
        shared_configuration_families,
    )

    print(
        "Cross-domain feasible families:",
        cross_domain_feasible_families,
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
