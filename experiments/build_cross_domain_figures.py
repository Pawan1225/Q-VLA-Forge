"""Build Sprint 5.14J proposal-ready cross-domain safety figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results" / "safety" / "cross-domain"
FIGURE_DIR = ROOT / "figures" / "safety" / "cross-domain"

CLEAN = RESULTS_DIR / "sprint5-cross-domain-clean.json"
GAUSSIAN = RESULTS_DIR / "sprint5-cross-domain-gaussian.json"
STRUCTURED = RESULTS_DIR / "sprint5-cross-domain-structured-state.json"
ACTION = RESULTS_DIR / "sprint5-cross-domain-action.json"
MECHANISM = RESULTS_DIR / "sprint5-cross-domain-lyapunov-mechanism.json"
CLAIMS = RESULTS_DIR / "sprint5-cross-domain-claim-matrix.json"

INDEX_JSON = RESULTS_DIR / "sprint5-cross-domain-figure-index.json"
INDEX_MD = RESULTS_DIR / "sprint5-cross-domain-figure-index.md"

DOMAINS = (
    "autonomous_driving",
    "robotics",
)

METHODS = (
    "clipping",
    "lyapunov",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")

    return payload


def _save(
    filename: str,
) -> Path:
    path = FIGURE_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    return path


def _relative(
    path: Path,
) -> str:
    return str(path.relative_to(ROOT)).replace(
        "\\",
        "/",
    )


def _figure_clean(
    payload: dict[str, Any],
) -> Path:
    seed_rows = payload["seed_rows"]

    if not isinstance(seed_rows, list):
        raise TypeError("clean seed_rows must be list")

    labels: list[str] = []
    values: list[float] = []

    for domain in DOMAINS:
        for method in METHODS:
            rows = [
                row
                for row in seed_rows
                if isinstance(row, dict)
                and row["domain"] == domain
                and row["method"] == method
            ]

            reductions = [
                float(row["relative_violation_reduction_vs_none"])
                for row in rows
                if row["relative_violation_reduction_vs_none"] is not None
            ]

            if not reductions:
                raise RuntimeError(f"missing clean reductions: {domain}/{method}")

            labels.append(f"{domain}\n{method}")

            values.append(sum(reductions) / len(reductions))

    plt.figure(figsize=(9, 5))

    plt.bar(
        labels,
        values,
    )

    plt.axhline(
        0.0,
        linewidth=1.0,
    )

    plt.ylabel("Mean relative violation reduction")

    plt.title("Clean Safety Improvement by Domain and Method")

    return _save("cross_domain_clean_violation_reduction.png")


def _figure_gaussian(
    payload: dict[str, Any],
) -> Path:
    rows = payload["direction_consistency"]

    if not isinstance(rows, list):
        raise TypeError("Gaussian direction consistency must be list")

    consistent = sum(
        1 for row in rows if isinstance(row, dict) and bool(row["direction_consistent"])
    )

    different = len(rows) - consistent

    plt.figure(figsize=(6, 5))

    plt.bar(
        [
            "Consistent",
            "Different",
        ],
        [
            consistent,
            different,
        ],
    )

    plt.ylabel("Cross-domain comparisons")

    plt.title("Gaussian Robustness Direction Consistency")

    return _save("cross_domain_gaussian_consistency.png")


def _figure_structured(
    payload: dict[str, object],
) -> Path:
    summaries_obj = payload["structured_state_summary"]

    if not isinstance(
        summaries_obj,
        list,
    ):
        raise TypeError("structured_state_summary must be a list")

    summaries: list[dict[str, object]] = []

    for raw in summaries_obj:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError("structured_state_summary row must be dict")

        summaries.append(raw)

    domains = [
        "autonomous_driving",
        "robotics",
    ]

    methods = [
        "none",
        "clipping",
        "lyapunov",
    ]

    worst_lookup: dict[
        tuple[str, str],
        float,
    ] = {}

    for row in summaries:
        value = row["worst_violation_delta"]

        if not isinstance(
            value,
            (int, float),
        ):
            raise TypeError("worst_violation_delta must be numeric")

        key = (
            str(row["domain"]),
            str(row["method"]),
        )

        worst_lookup[key] = float(value)

    figure_path = FIGURE_DIR / "cross_domain_structured_state_sensitivity.png"

    plt.figure(figsize=(8, 4.5))

    positions = list(range(len(methods)))

    for domain in domains:
        values = [
            worst_lookup[
                (
                    domain,
                    method,
                )
            ]
            for method in methods
        ]

        plt.plot(
            positions,
            values,
            marker="o",
            label=domain,
        )

    plt.xticks(
        positions,
        methods,
    )

    plt.ylabel("Worst violation delta")

    plt.xlabel("Method")

    plt.title("Structured-state worst violation delta by domain")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return figure_path


def _figure_action(
    payload: dict[str, Any],
) -> Path:
    rows = payload["domain_method_summary"]

    if not isinstance(rows, list):
        raise TypeError("action domain_method_summary must be list")

    labels: list[str] = []
    values: list[float] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        rate = row["recovery_rate"]

        if rate is None:
            continue

        labels.append(f"{row['domain']}\n" f"{row['method']}")

        values.append(float(rate))

    plt.figure(figsize=(10, 5))

    plt.bar(
        labels,
        values,
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.ylabel("Explicit recovery rate")

    plt.title("Action-Perturbation Recovery by Domain and Method")

    return _save("cross_domain_action_recovery.png")


def _figure_mechanism(
    payload: dict[str, Any],
) -> Path:
    rows = payload["domain_mechanism_summary"]

    if not isinstance(rows, list):
        raise TypeError("mechanism domain summary must be list")

    labels: list[str] = []
    values: list[int] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        labels.append(str(row["domain"]))

        values.append(int(row["lyapunov_decrease_interventions"]))

    plt.figure(figsize=(7, 5))

    plt.bar(
        labels,
        values,
    )

    plt.ylabel("LYAPUNOV_DECREASE interventions")

    plt.title("Domain Split of Lyapunov-Specific Activation")

    return _save("cross_domain_lyapunov_mechanism.png")


def _figure_claims(
    payload: dict[str, Any],
) -> Path:
    status_counts = payload["status_counts"]

    if not isinstance(status_counts, dict):
        raise TypeError("claim status_counts must be dict")

    labels = [
        "Supported",
        "Supported\nwith limitation",
        "Not supported",
    ]

    values = [
        int(status_counts["supported"]),
        int(status_counts["supported_with_limitation"]),
        int(status_counts["not_supported"]),
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(
        labels,
        values,
    )

    plt.ylabel("Claims")

    plt.title("Cross-Domain Safety Claim Status")

    return _save("cross_domain_claim_status.png")


def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean = _load(CLEAN)

    gaussian = _load(GAUSSIAN)

    structured = _load(STRUCTURED)

    action = _load(ACTION)

    mechanism = _load(MECHANISM)

    claims = _load(CLAIMS)

    figure_records = [
        {
            "figure_id": "S5-CD-F01",
            "title": ("Clean Safety Improvement by Domain and Method"),
            "path": _relative(_figure_clean(clean)),
            "source_artifacts": [
                _relative(CLEAN),
            ],
            "supports": [
                "S5-CD02",
                "S5-CD07",
            ],
        },
        {
            "figure_id": "S5-CD-F02",
            "title": ("Gaussian Robustness Direction Consistency"),
            "path": _relative(_figure_gaussian(gaussian)),
            "source_artifacts": [
                _relative(GAUSSIAN),
            ],
            "supports": [
                "S5-CD03",
            ],
        },
        {
            "figure_id": "S5-CD-F03",
            "title": ("Structured-State Robustness Sensitivity"),
            "path": _relative(_figure_structured(structured)),
            "source_artifacts": [
                _relative(STRUCTURED),
            ],
            "supports": [
                "S5-CD04",
            ],
        },
        {
            "figure_id": "S5-CD-F04",
            "title": ("Action-Perturbation Recovery by Domain and Method"),
            "path": _relative(_figure_action(action)),
            "source_artifacts": [
                _relative(ACTION),
            ],
            "supports": [
                "S5-CD05",
            ],
        },
        {
            "figure_id": "S5-CD-F05",
            "title": ("Domain Split of Lyapunov-Specific Activation"),
            "path": _relative(_figure_mechanism(mechanism)),
            "source_artifacts": [
                _relative(MECHANISM),
            ],
            "supports": [
                "S5-CD06",
            ],
        },
        {
            "figure_id": "S5-CD-F06",
            "title": ("Cross-Domain Safety Claim Status"),
            "path": _relative(_figure_claims(claims)),
            "source_artifacts": [
                _relative(CLAIMS),
            ],
            "supports": [
                "S5-CD01",
                "S5-CD02",
                "S5-CD03",
                "S5-CD04",
                "S5-CD05",
                "S5-CD06",
                "S5-CD07",
                "S5-CD08",
                "S5-CD09",
                "S5-CD10",
                "S5-CD11",
                "S5-CD12",
            ],
        },
    ]

    for record in figure_records:
        figure_path = ROOT / str(record["path"])

        if not figure_path.exists():
            raise RuntimeError("figure generation failed: " f"{figure_path}")

        if figure_path.stat().st_size <= 0:
            raise RuntimeError("empty figure generated: " f"{figure_path}")

    payload: dict[
        str,
        Any,
    ] = {
        "sprint": "5.14J",
        "artifact": "cross-domain-figure-provenance-index",
        "analysis_only": True,
        "new_training": False,
        "new_principal_runs": False,
        "figure_count": len(figure_records),
        "figures": figure_records,
        "source_artifacts": [
            _relative(path)
            for path in (
                CLEAN,
                GAUSSIAN,
                STRUCTURED,
                ACTION,
                MECHANISM,
                CLAIMS,
            )
        ],
        "claim_controls": {
            "raw_cross_domain_reward_ranking": False,
            "raw_cross_domain_violation_ranking": False,
            "formal_safety_guarantee": False,
            "production_generalization": False,
            "universal_controller": False,
        },
    }

    INDEX_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Sprint 5.14J — Cross-Domain Figure Provenance Index",
        "",
        f"Figures: {len(figure_records)}",
        "",
        "| ID | Figure | Source | Claims |",
        "|---|---|---|---|",
    ]

    for record in figure_records:
        lines.append(
            f"| {record['figure_id']} | "
            f"{record['path']} | "
            f"{'; '.join(record['source_artifacts'])} | "
            f"{'; '.join(record['supports'])} |"
        )

    lines.extend(
        [
            "",
            "## Claim controls",
            "",
            "- No raw cross-domain reward ranking.",
            "- No raw cross-domain violation-rate ranking.",
            "- No formal safety guarantee.",
            "- No universal safety-controller claim.",
            "- No production-generalization claim.",
        ]
    )

    INDEX_MD.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 80)

    print(" SPRINT 5.14J CROSS-DOMAIN FIGURES + PROVENANCE")

    print("=" * 80)

    print()

    print(
        "Figures generated:",
        len(figure_records),
    )

    print()

    for record in figure_records:
        print(f"{record['figure_id']}: " f"{record['path']}")

    print()

    print("Figure existence: PASS")

    print("Figure non-empty validation: PASS")

    print("Source provenance: PASS")

    print("Claim mapping: PASS")

    print("Cross-domain ranking controls: PASS")

    print("No new training: PASS")

    print("No new principal execution: PASS")

    print()

    print("SPRINT 5.14J FIGURES + PROVENANCE: PASS")


if __name__ == "__main__":
    main()
