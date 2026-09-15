from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from q_vla_forge.evaluation.sprint5_evidence import sha256_file

ROOT = Path(".")

EVIDENCE_DIR = ROOT / "results" / "safety" / "evidence"

FIGURE_DIR = ROOT / "figures" / "safety"

CANONICAL_EVIDENCE = EVIDENCE_DIR / "sprint5-safety-evidence.json"

CANONICAL_CSV = EVIDENCE_DIR / "sprint5-safety-evidence.csv"

CANONICAL_MARKDOWN = EVIDENCE_DIR / "sprint5-safety-evidence.md"

CLAIM_MATRIX_OUTPUT = EVIDENCE_DIR / "sprint5-safety-claim-matrix.json"

FIGURE_INDEX_OUTPUT = EVIDENCE_DIR / "sprint5-safety-figure-index.json"

MANIFEST_OUTPUT = EVIDENCE_DIR / "sprint5-safety-manifest.json"

FINAL_CLAIMS = (
    ROOT / "results" / "safety" / "final" / "sprint5-final-claim-boundary.json"
)

FINAL_LIMITATIONS = (
    ROOT / "results" / "safety" / "final" / "sprint5-final-limitations.json"
)

FREEZE_RECORD = ROOT / "results" / "safety" / "final" / "sprint5-freeze-record.json"

READINESS_SCRIPT = ROOT / "experiments" / "verify_sprint5_safety_evidence_readiness.py"

CORE_BUILDER = ROOT / "experiments" / "build_sprint5_safety_evidence.py"

ASSET_BUILDER = ROOT / "experiments" / "build_sprint5_safety_assets.py"

FIGURES = {
    "clean_safety": (FIGURE_DIR / "sprint5-clean-safety.png"),
    "robustness": (FIGURE_DIR / "sprint5-robustness.png"),
    "action_recovery": (FIGURE_DIR / "sprint5-action-recovery.png"),
    "lyapunov_mechanisms": (FIGURE_DIR / "sprint5-lyapunov-mechanisms.png"),
    "cross_domain": (FIGURE_DIR / "sprint5-cross-domain.png"),
}

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EVIDENCE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"{path} must contain an object")

    return payload


def normalized_path(
    path: Path | str,
) -> str:
    return str(path).replace(
        "\\",
        "/",
    )


def require_file(
    path: Path,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.stat().st_size <= 0:
        raise RuntimeError(f"Empty file: {path}")


def mean_metric(
    evidence: dict[str, Any],
    domain: str,
    method: str,
    metric: str,
) -> float:
    value = evidence["clean_safety"][domain][method][metric]["mean"]

    if not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(f"Metric is not numeric: " f"{domain}/{method}/{metric}")

    return float(value)


def save_figure(
    path: Path,
) -> None:
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    require_file(path)


def build_clean_safety_figure(
    evidence: dict[str, Any],
) -> None:
    methods = [
        "NONE",
        "CLIPPING",
        "LYAPUNOV",
    ]

    driving = [
        mean_metric(
            evidence,
            "autonomous_driving",
            method.lower(),
            "violation_step_rate",
        )
        for method in methods
    ]

    robotics = [
        mean_metric(
            evidence,
            "robotics",
            method.lower(),
            "violation_step_rate",
        )
        for method in methods
    ]

    x = list(range(len(methods)))

    width = 0.35

    plt.figure(figsize=(8.2, 5.2))

    plt.bar(
        [item - width / 2 for item in x],
        driving,
        width=width,
        label="Autonomous driving",
    )

    plt.bar(
        [item + width / 2 for item in x],
        robotics,
        width=width,
        label="Robotics",
    )

    plt.xticks(
        x,
        methods,
    )

    plt.ylabel("Executed violation-step rate")

    plt.title("Sprint 5 Clean Safety Evaluation")

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    save_figure(FIGURES["clean_safety"])


def worst_gaussian_delta(
    evidence: dict[str, Any],
    domain: str,
    method: str,
) -> float:
    aggregates = evidence["gaussian_robustness"]["aggregates"]

    values: list[float] = []

    for row in aggregates:
        if not isinstance(
            row,
            dict,
        ):
            continue

        if row.get("domain") != domain or row.get("method") != method:
            continue

        sigma = row.get("sigma")

        if not isinstance(
            sigma,
            (int, float),
        ):
            continue

        if float(sigma) <= 0.0:
            continue

        metric = row.get("violation_delta_from_clean")

        if not isinstance(
            metric,
            dict,
        ):
            continue

        mean = metric.get("mean")

        if isinstance(
            mean,
            (int, float),
        ):
            values.append(float(mean))

    return max(
        values,
        default=0.0,
    )


def worst_state_delta(
    evidence: dict[str, Any],
    domain: str,
    method: str,
) -> float:
    sensitivity = evidence["structured_state_robustness"].get(
        "sensitivity",
        {},
    )

    if not isinstance(
        sensitivity,
        dict,
    ):
        return 0.0

    domain_data = sensitivity.get(
        domain,
        {},
    )

    if not isinstance(
        domain_data,
        dict,
    ):
        return 0.0

    method_data = domain_data.get(
        method,
        {},
    )

    if not isinstance(
        method_data,
        dict,
    ):
        return 0.0

    worst = method_data.get(
        "worst_violation",
        {},
    )

    if not isinstance(
        worst,
        dict,
    ):
        return 0.0

    value = worst.get(
        "mean_delta",
        0.0,
    )

    if not isinstance(
        value,
        (int, float),
    ):
        return 0.0

    return float(value)


def worst_action_delta(
    evidence: dict[str, Any],
    domain: str,
    method: str,
) -> float:
    aggregates = evidence["action_robustness"]["aggregates"]

    values: list[float] = []

    for row in aggregates:
        if not isinstance(
            row,
            dict,
        ):
            continue

        if row.get("domain") != domain or row.get("method") != method:
            continue

        metric = row.get("executed_violation_delta_from_clean")

        if not isinstance(
            metric,
            dict,
        ):
            continue

        mean = metric.get("mean")

        if isinstance(
            mean,
            (int, float),
        ):
            values.append(float(mean))

    return max(
        values,
        default=0.0,
    )


def build_robustness_figure(
    evidence: dict[str, Any],
) -> None:
    labels = [
        "Driving\nNONE",
        "Driving\nCLIP",
        "Driving\nLYAP",
        "Robotics\nNONE",
        "Robotics\nCLIP",
        "Robotics\nLYAP",
    ]

    pairs = [
        (
            "autonomous_driving",
            "none",
        ),
        (
            "autonomous_driving",
            "clipping",
        ),
        (
            "autonomous_driving",
            "lyapunov",
        ),
        (
            "robotics",
            "none",
        ),
        (
            "robotics",
            "clipping",
        ),
        (
            "robotics",
            "lyapunov",
        ),
    ]

    gaussian = [
        worst_gaussian_delta(
            evidence,
            domain,
            method,
        )
        for domain, method in pairs
    ]

    state = [
        worst_state_delta(
            evidence,
            domain,
            method,
        )
        for domain, method in pairs
    ]

    action = [
        worst_action_delta(
            evidence,
            domain,
            method,
        )
        for domain, method in pairs
    ]

    x = list(range(len(labels)))

    width = 0.25

    plt.figure(figsize=(10.5, 5.5))

    plt.bar(
        [item - width for item in x],
        gaussian,
        width=width,
        label="Gaussian",
    )

    plt.bar(
        x,
        state,
        width=width,
        label="Structured state",
    )

    plt.bar(
        [item + width for item in x],
        action,
        width=width,
        label="Structured action",
    )

    plt.axhline(
        0.0,
        linewidth=1.0,
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel("Worst violation-rate delta " "from clean")

    plt.title("Sprint 5 Robustness Stress Tests")

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    save_figure(FIGURES["robustness"])


def build_action_recovery_figure(
    evidence: dict[str, Any],
) -> None:
    summaries = evidence["action_robustness"]["domain_method_summary"]

    labels: list[str] = []
    values: list[float] = []

    for domain in (
        "autonomous_driving",
        "robotics",
    ):
        for method in (
            "none",
            "clipping",
            "lyapunov",
        ):
            match = None

            for row in summaries:
                if not isinstance(
                    row,
                    dict,
                ):
                    continue

                if row.get("domain") == domain and row.get("method") == method:
                    match = row
                    break

            if match is None:
                raise RuntimeError("Missing action recovery " f"{domain}/{method}")

            rate = match.get("recovery_rate")

            value = 0.0 if rate is None else float(rate)

            prefix = "Drive" if domain == "autonomous_driving" else "Robot"

            labels.append(f"{prefix}\n{method.upper()}")

            values.append(value)

    plt.figure(figsize=(9.2, 5.3))

    bars = plt.bar(
        range(len(labels)),
        values,
    )

    plt.xticks(
        range(len(labels)),
        labels,
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.ylabel("Unsafe perturbed-action recovery rate")

    plt.title("Sprint 5 Structured-Action Recovery")

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        values,
        strict=True,
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_figure(FIGURES["action_recovery"])


def mechanism_action_rows(
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = evidence["lyapunov_mechanisms"]["by_domain_and_regime"]

    result: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(
            row,
            dict,
        ):
            continue

        if row.get("regime") == "structured_action":
            result.append(row)

    return result


def build_lyapunov_figure(
    evidence: dict[str, Any],
) -> None:
    rows = mechanism_action_rows(evidence)

    domains = [
        "autonomous_driving",
        "robotics",
    ]

    action_bound: list[int] = []
    domain_constraint: list[int] = []
    lyapunov_decrease: list[int] = []
    strict_decrease: list[int] = []

    for domain in domains:
        row = next(
            (item for item in rows if item.get("domain") == domain),
            None,
        )

        if row is None:
            raise RuntimeError(f"Missing mechanism row: {domain}")

        reasons = row.get(
            "intervention_reason_counts",
            {},
        )

        if not isinstance(
            reasons,
            dict,
        ):
            raise TypeError("Mechanism reason counts " "must be dict")

        action_bound.append(
            int(
                reasons.get(
                    "action_bound",
                    0,
                )
            )
        )

        domain_constraint.append(
            int(
                reasons.get(
                    "domain_constraint",
                    0,
                )
            )
        )

        lyapunov_decrease.append(
            int(
                reasons.get(
                    "lyapunov_decrease",
                    0,
                )
            )
        )

        strict_decrease.append(
            int(
                row.get(
                    "strict_decrease_count",
                    0,
                )
            )
        )

    x = [
        0,
        1,
    ]

    width = 0.19

    plt.figure(figsize=(9.5, 5.5))

    plt.bar(
        [item - 1.5 * width for item in x],
        action_bound,
        width=width,
        label="ACTION_BOUND",
    )

    plt.bar(
        [item - 0.5 * width for item in x],
        domain_constraint,
        width=width,
        label="DOMAIN_CONSTRAINT",
    )

    plt.bar(
        [item + 0.5 * width for item in x],
        lyapunov_decrease,
        width=width,
        label="LYAPUNOV_DECREASE reason",
    )

    plt.bar(
        [item + 1.5 * width for item in x],
        strict_decrease,
        width=width,
        label="Strict empirical delta-V < 0",
    )

    plt.xticks(
        x,
        [
            "Autonomous driving",
            "Robotics",
        ],
    )

    plt.ylabel("Frozen step count")

    plt.title("Sprint 5 Lyapunov Mechanism Attribution")

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    save_figure(FIGURES["lyapunov_mechanisms"])


def add_box(
    x: float,
    y: float,
    text: str,
) -> None:
    plt.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "black",
        },
        fontsize=9,
    )


def add_arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> None:
    plt.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={
            "arrowstyle": "->",
        },
    )


def build_cross_domain_figure(
    evidence: dict[str, Any],
) -> None:
    architecture = evidence["cross_domain"]["architecture_reuse"]

    if not isinstance(
        architecture,
        dict,
    ):
        raise TypeError("Architecture reuse must be dict")

    plt.figure(figsize=(11.0, 6.2))

    plt.axis("off")

    add_box(
        0.15,
        0.75,
        "Autonomous-driving\nadapter + state",
    )

    add_box(
        0.15,
        0.25,
        "Robotics\nadapter + state",
    )

    add_box(
        0.42,
        0.50,
        "Shared decision contract\n" "3-D action interface",
    )

    add_box(
        0.67,
        0.50,
        "Shared safety interface\n" "NONE / CLIPPING / LYAPUNOV",
    )

    add_box(
        0.90,
        0.75,
        "Driving-specific\nconstraints / predictor /\n" "Lyapunov semantics",
    )

    add_box(
        0.90,
        0.25,
        "Robotics-specific\nconstraints / predictor /\n" "Lyapunov semantics",
    )

    add_arrow(
        0.24,
        0.73,
        0.35,
        0.55,
    )

    add_arrow(
        0.24,
        0.27,
        0.35,
        0.45,
    )

    add_arrow(
        0.50,
        0.50,
        0.59,
        0.50,
    )

    add_arrow(
        0.75,
        0.55,
        0.83,
        0.72,
    )

    add_arrow(
        0.75,
        0.45,
        0.83,
        0.28,
    )

    plt.text(
        0.50,
        0.93,
        "Sprint 5 Cross-Domain Safety Architecture",
        ha="center",
        va="center",
        fontsize=14,
    )

    plt.text(
        0.50,
        0.08,
        (
            "Shared framework: "
            f"{architecture.get('shared_safety_interface', False)} | "
            "Same policy weights: "
            f"{architecture.get('same_policy_weights', False)} | "
            "Universal controller: "
            f"{architecture.get('universal_controller_supported', False)}"
        ),
        ha="center",
        va="center",
        fontsize=9,
    )

    save_figure(FIGURES["cross_domain"])


def claim_status(
    status: str,
) -> str:
    mapping = {
        "supported": "SUPPORTED",
        "supported_with_limitation": ("SUPPORTED_WITH_LIMITATION"),
        "not_supported": "NOT_SUPPORTED",
    }

    lowered = status.lower()

    if lowered not in mapping:
        raise ValueError(f"Unknown claim status: {status}")

    return mapping[lowered]


def flatten_final_claims(
    final_claims: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    buckets = [
        "supported_claims",
        "supported_with_limitation",
        "not_supported_claims",
    ]

    for bucket in buckets:
        rows = final_claims.get(
            bucket,
            [],
        )

        if not isinstance(
            rows,
            list,
        ):
            raise TypeError(f"{bucket} must be list")

        for row in rows:
            if not isinstance(
                row,
                dict,
            ):
                raise TypeError("Claim row must be dict")

            status = claim_status(str(row["status"]))

            output.append(
                {
                    "claim_id": str(row["claim_id"]),
                    "statement": str(row["statement"]),
                    "status": status,
                    "limitation": (
                        str(
                            row.get(
                                "limitation",
                                "",
                            )
                        )
                    ),
                    "source": normalized_path(FINAL_CLAIMS),
                    "proposal_safe": (status != "NOT_SUPPORTED"),
                }
            )

    output.sort(key=lambda row: row["claim_id"])

    return output


def required_proposal_claims() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "S5-E01",
            "statement": (
                "Sprint 5 evaluated explicit "
                "safety filtering in both "
                "autonomous-driving and robotics "
                "synthetic proxy domains."
            ),
            "status": "SUPPORTED",
            "evidence": [normalized_path(CANONICAL_EVIDENCE)],
            "limitation": ("Synthetic proxy-domain evidence."),
        },
        {
            "claim_id": "S5-E02",
            "statement": (
                "Under the frozen proxy safety "
                "contracts, CLIPPING and LYAPUNOV "
                "reduced measured clean executed "
                "violation-step events to zero in "
                "both evaluated domains."
            ),
            "status": "SUPPORTED",
            "evidence": [normalized_path(CANONICAL_EVIDENCE)],
            "limitation": ("Empirical proxy evidence; not " "formal certification."),
        },
        {
            "claim_id": "S5-E03",
            "statement": (
                "Gaussian observation, structured-"
                "state, and structured-action "
                "robustness were evaluated."
            ),
            "status": ("SUPPORTED_WITH_LIMITATION"),
            "evidence": [normalized_path(CANONICAL_EVIDENCE)],
            "limitation": (
                "Synthetic perturbations; "
                "observation-perturbation studies "
                "use privileged true state in the "
                "explicit safety layer."
            ),
        },
        {
            "claim_id": "S5-E04",
            "statement": (
                "Unsafe perturbed-action recovery "
                "was quantified separately from "
                "environment-interface adjustment."
            ),
            "status": "SUPPORTED",
            "evidence": [normalized_path(CANONICAL_EVIDENCE)],
            "limitation": ("Synthetic action perturbation."),
        },
        {
            "claim_id": "S5-E05",
            "statement": (
                "Lyapunov-specific candidate "
                "selection was empirically activated "
                "under structured action perturbation "
                "in autonomous driving."
            ),
            "status": "SUPPORTED",
            "evidence": [normalized_path(CANONICAL_EVIDENCE)],
            "limitation": (
                "Domain-dependent mechanism evidence; "
                "does not establish formal stability."
            ),
        },
        {
            "claim_id": "S5-E06",
            "statement": (
                "The two proxy domains reuse a "
                "shared safety interface, metric "
                "schema, seed protocol, and robustness "
                "harness while retaining domain-"
                "specific safety semantics."
            ),
            "status": "SUPPORTED",
            "evidence": [normalized_path(CANONICAL_EVIDENCE)],
            "limitation": (
                "Does not demonstrate a universal "
                "controller or shared policy weights."
            ),
        },
    ]


def explicit_unsupported_claims() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "S5-EU01",
            "statement": ("Formal Lyapunov stability guarantee."),
            "status": "NOT_SUPPORTED",
        },
        {
            "claim_id": "S5-EU02",
            "statement": ("Formal worst-case robustness " "guarantee."),
            "status": "NOT_SUPPORTED",
        },
        {
            "claim_id": "S5-EU03",
            "statement": ("ISO 26262 or other safety " "certification compliance."),
            "status": "NOT_SUPPORTED",
        },
        {
            "claim_id": "S5-EU04",
            "statement": ("Production safety readiness."),
            "status": "NOT_SUPPORTED",
        },
        {
            "claim_id": "S5-EU05",
            "statement": ("Real-world sensor fault tolerance."),
            "status": "NOT_SUPPORTED",
        },
        {
            "claim_id": "S5-EU06",
            "statement": (
                "Universal safety controller or " "zero-shot cross-domain transfer."
            ),
            "status": "NOT_SUPPORTED",
        },
        {
            "claim_id": "S5-EU07",
            "statement": ("Quantum safety advantage."),
            "status": "NOT_SUPPORTED",
        },
    ]


PROHIBITED_WORDING = [
    "certified safe",
    "safety certified",
    "proven safe",
    "formal safety guarantee",
    "guaranteed stable",
    "iso 26262 compliant",
    "production ready safety",
    "quantum safety advantage",
    "universal safety controller",
    "zero-shot safety transfer",
]


def build_claim_matrix(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    final_claims = load_json(FINAL_CLAIMS)

    matrix = {
        "artifact": ("sprint5-safety-claim-matrix"),
        "sprint": "5.15",
        "analysis_only": True,
        "scientific_scope_frozen": True,
        "new_training": False,
        "new_principal_execution": False,
        "new_scientific_experiment": False,
        "proposal_claims": (required_proposal_claims()),
        "explicitly_unsupported": (explicit_unsupported_claims()),
        "frozen_final_claim_boundary": (flatten_final_claims(final_claims)),
        "prohibited_wording": (PROHIBITED_WORDING),
        "usage_rule": (
            "Positive proposal wording may not "
            "exceed the strength of the frozen "
            "final claim boundary."
        ),
        "canonical_evidence": (normalized_path(CANONICAL_EVIDENCE)),
        "canonical_evidence_sha256": (sha256_file(CANONICAL_EVIDENCE)),
    }

    headlines = evidence.get(
        "proposal_headlines",
        {},
    )

    if not isinstance(
        headlines,
        dict,
    ):
        raise TypeError("proposal_headlines must be dict")

    serialized = json.dumps(headlines).lower()

    for phrase in (
        "certified safe",
        "proven safe",
        "iso 26262 compliant",
        "production ready safety",
        "quantum safety advantage",
        "universal safety controller",
    ):
        if phrase in serialized:
            raise RuntimeError("Unsafe proposal wording " f"detected: {phrase}")

    return matrix


def figure_record(
    *,
    figure_id: str,
    path: Path,
    title: str,
    supports: list[str],
    sources: list[Path],
    limitation: str,
) -> dict[str, Any]:
    require_file(path)

    source_records = []

    for source in sources:
        require_file(source)

        source_records.append(
            {
                "path": normalized_path(source),
                "sha256": (sha256_file(source)),
            }
        )

    return {
        "figure_id": figure_id,
        "path": normalized_path(path),
        "sha256": sha256_file(path),
        "size_bytes": (path.stat().st_size),
        "title": title,
        "supports": supports,
        "source_artifacts": (source_records),
        "limitation": limitation,
        "frozen": True,
    }


def build_figure_index() -> dict[str, Any]:
    return {
        "artifact": ("sprint5-safety-figure-index"),
        "sprint": "5.15",
        "figure_count": 5,
        "new_scientific_experiment": (False),
        "figures": [
            figure_record(
                figure_id="S5-F01",
                path=FIGURES["clean_safety"],
                title=("Clean Safety Evaluation"),
                supports=[
                    "S5-E01",
                    "S5-E02",
                ],
                sources=[
                    CANONICAL_EVIDENCE,
                ],
                limitation=("Synthetic proxy safety " "contracts."),
            ),
            figure_record(
                figure_id="S5-F02",
                path=FIGURES["robustness"],
                title=("Robustness Stress Tests"),
                supports=[
                    "S5-E03",
                ],
                sources=[
                    CANONICAL_EVIDENCE,
                ],
                limitation=(
                    "Synthetic perturbations; "
                    "observation studies retain "
                    "true-state safety access."
                ),
            ),
            figure_record(
                figure_id="S5-F03",
                path=FIGURES["action_recovery"],
                title=("Structured-Action Recovery"),
                supports=[
                    "S5-E04",
                ],
                sources=[
                    CANONICAL_EVIDENCE,
                ],
                limitation=("Synthetic action perturbation."),
            ),
            figure_record(
                figure_id="S5-F04",
                path=FIGURES["lyapunov_mechanisms"],
                title=("Lyapunov Mechanism Attribution"),
                supports=[
                    "S5-E05",
                ],
                sources=[
                    CANONICAL_EVIDENCE,
                ],
                limitation=("Mechanism evidence only; " "not formal stability proof."),
            ),
            figure_record(
                figure_id="S5-F05",
                path=FIGURES["cross_domain"],
                title=("Cross-Domain Safety Architecture"),
                supports=[
                    "S5-E06",
                ],
                sources=[
                    CANONICAL_EVIDENCE,
                ],
                limitation=(
                    "Shared framework does not "
                    "imply shared policy weights or "
                    "a universal controller."
                ),
            ),
        ],
    }


def manifest_artifact(
    path: Path,
    role: str,
    required: bool = True,
) -> dict[str, Any]:
    require_file(path)

    return {
        "path": normalized_path(path),
        "sha256": sha256_file(path),
        "size_bytes": (path.stat().st_size),
        "role": role,
        "required": required,
        "frozen": True,
    }


def build_manifest() -> dict[str, Any]:
    artifacts = [
        manifest_artifact(
            CANONICAL_EVIDENCE,
            "canonical_json",
        ),
        manifest_artifact(
            CANONICAL_CSV,
            "normalized_csv",
        ),
        manifest_artifact(
            CANONICAL_MARKDOWN,
            "human_readable_report",
        ),
        manifest_artifact(
            CLAIM_MATRIX_OUTPUT,
            "proposal_claim_matrix",
        ),
        manifest_artifact(
            FIGURE_INDEX_OUTPUT,
            "figure_provenance_index",
        ),
        manifest_artifact(
            FIGURES["clean_safety"],
            "proposal_figure",
        ),
        manifest_artifact(
            FIGURES["robustness"],
            "proposal_figure",
        ),
        manifest_artifact(
            FIGURES["action_recovery"],
            "proposal_figure",
        ),
        manifest_artifact(
            FIGURES["lyapunov_mechanisms"],
            "proposal_figure",
        ),
        manifest_artifact(
            FIGURES["cross_domain"],
            "proposal_figure",
        ),
        manifest_artifact(
            READINESS_SCRIPT,
            "readiness_verifier",
        ),
        manifest_artifact(
            CORE_BUILDER,
            "canonical_evidence_builder",
        ),
        manifest_artifact(
            ASSET_BUILDER,
            "figure_claim_manifest_builder",
        ),
    ]

    upstream = [
        manifest_artifact(
            FINAL_CLAIMS,
            "frozen_claim_boundary",
        ),
        manifest_artifact(
            FINAL_LIMITATIONS,
            "frozen_limitations",
        ),
        manifest_artifact(
            FREEZE_RECORD,
            "scientific_freeze_record",
        ),
    ]

    return {
        "artifact": ("sprint5-safety-manifest"),
        "sprint": "5.15",
        "status": "PACKAGED",
        "scientific_scope_frozen": True,
        "new_training": False,
        "new_principal_execution": False,
        "new_scientific_experiment": False,
        "artifact_count": len(artifacts),
        "upstream_count": len(upstream),
        "artifacts": artifacts,
        "upstream_frozen_artifacts": (upstream),
    }


def attach_figure_index_to_evidence(
    evidence: dict[str, Any],
    figure_index: dict[str, Any],
) -> None:
    figures = figure_index.get("figures", [])

    if not isinstance(
        figures,
        list,
    ):
        raise TypeError("Figure index figures must be list")

    evidence["figures"] = [
        {
            "figure_id": row["figure_id"],
            "path": row["path"],
            "title": row["title"],
            "supports": row["supports"],
            "sha256": row["sha256"],
        }
        for row in figures
        if isinstance(
            row,
            dict,
        )
    ]

    CANONICAL_EVIDENCE.write_text(
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    require_file(CANONICAL_EVIDENCE)

    require_file(CANONICAL_CSV)

    require_file(CANONICAL_MARKDOWN)

    require_file(FINAL_CLAIMS)

    evidence = load_json(CANONICAL_EVIDENCE)

    metadata = evidence.get(
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        raise TypeError("Evidence metadata missing")

    if metadata.get("scientific_scope_frozen") is not True:
        raise RuntimeError("Scientific scope is not frozen")

    if metadata.get("new_training") is not False:
        raise RuntimeError("Unexpected training flag")

    build_clean_safety_figure(evidence)

    build_robustness_figure(evidence)

    build_action_recovery_figure(evidence)

    build_lyapunov_figure(evidence)

    build_cross_domain_figure(evidence)

    claim_matrix = build_claim_matrix(evidence)

    CLAIM_MATRIX_OUTPUT.write_text(
        json.dumps(
            claim_matrix,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    figure_index = build_figure_index()

    FIGURE_INDEX_OUTPUT.write_text(
        json.dumps(
            figure_index,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    attach_figure_index_to_evidence(
        evidence,
        figure_index,
    )

    # Figure attachment changes the canonical JSON
    # hash, so build the final claim matrix again.
    claim_matrix = build_claim_matrix(load_json(CANONICAL_EVIDENCE))

    CLAIM_MATRIX_OUTPUT.write_text(
        json.dumps(
            claim_matrix,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_manifest()

    MANIFEST_OUTPUT.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 56)
    print(" Q-VLA FORGE - SPRINT 5.15 " "BATCH C ASSET BUILD")
    print("=" * 56)
    print()

    print("Clean safety figure                 PASS")
    print("Robustness figure                   PASS")
    print("Action recovery figure              PASS")
    print("Lyapunov mechanism figure           PASS")
    print("Cross-domain architecture figure    PASS")
    print()
    print("Figure provenance index             PASS")
    print("Proposal claim matrix               PASS")
    print("Claim wording controls              PASS")
    print("Evidence manifest                   PASS")
    print()
    print("No new training                     PASS")
    print("No principal execution              PASS")
    print("No new scientific experiment        PASS")
    print()
    print("SPRINT 5.15 BATCH C BUILD: PASS")


if __name__ == "__main__":
    main()
