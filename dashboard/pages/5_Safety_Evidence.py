from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_PATH = (
    ROOT / "results" / "safety" / "evidence" / "sprint5-safety-evidence.json"
)

CLAIM_MATRIX_PATH = (
    ROOT / "results" / "safety" / "evidence" / "sprint5-safety-claim-matrix.json"
)

FIGURE_INDEX_PATH = (
    ROOT / "results" / "safety" / "evidence" / "sprint5-safety-figure-index.json"
)

CSV_PATH = ROOT / "results" / "safety" / "evidence" / "sprint5-safety-evidence.csv"

MARKDOWN_PATH = ROOT / "results" / "safety" / "evidence" / "sprint5-safety-evidence.md"

MANIFEST_PATH = (
    ROOT / "results" / "safety" / "evidence" / "sprint5-safety-manifest.json"
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(f"{path} must contain a JSON object")

    return payload


def mean_value(
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
        raise TypeError(f"Non-numeric metric: " f"{domain}/{method}/{metric}")

    return float(value)


def fmt(
    value: float,
) -> str:
    return f"{value:.4f}"


def clean_rows(
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for domain, label in (
        (
            "autonomous_driving",
            "Autonomous Driving",
        ),
        (
            "robotics",
            "Robotics",
        ),
    ):
        for method in (
            "none",
            "clipping",
            "lyapunov",
        ):
            record = evidence["clean_safety"][domain][method]

            rows.append(
                {
                    "Domain": label,
                    "Method": (method.upper()),
                    "Violation-step rate": (
                        fmt(float(record["violation_step_rate"]["mean"]))
                    ),
                    "Reward": fmt(float(record["reward"]["mean"])),
                    "Success rate": fmt(float(record["success_rate"]["mean"])),
                    "Intervention rate": (
                        fmt(float(record["intervention_rate"]["mean"]))
                    ),
                }
            )

    return rows


def action_rows(
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    source = evidence["action_robustness"]["domain_method_summary"]

    if not isinstance(
        source,
        list,
    ):
        raise TypeError("Action summary must be list")

    rows: list[dict[str, Any]] = []

    for item in source:
        if not isinstance(
            item,
            dict,
        ):
            continue

        rate = item.get("recovery_rate")

        rows.append(
            {
                "Domain": (
                    "Autonomous Driving"
                    if item["domain"] == "autonomous_driving"
                    else "Robotics"
                ),
                "Method": str(item["method"]).upper(),
                "Unsafe": int(item["unsafe_perturbed_steps"]),
                "Recovered": int(item["recovered_unsafe_steps"]),
                "Unresolved": int(item["unresolved_unsafe_steps"]),
                "Recovery rate": ("Undefined" if rate is None else fmt(float(rate))),
            }
        )

    return rows


def mechanism_rows(
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    source = evidence["lyapunov_mechanisms"]["by_domain_and_regime"]

    if not isinstance(
        source,
        list,
    ):
        raise TypeError("Mechanism evidence must be list")

    rows: list[dict[str, Any]] = []

    for item in source:
        if not isinstance(
            item,
            dict,
        ):
            continue

        reasons = item.get(
            "intervention_reason_counts",
            {},
        )

        if not isinstance(
            reasons,
            dict,
        ):
            raise TypeError("Mechanism reason counts " "must be dict")

        rows.append(
            {
                "Domain": (
                    "Autonomous Driving"
                    if item["domain"] == "autonomous_driving"
                    else "Robotics"
                ),
                "Regime": str(item["regime"]),
                "ACTION_BOUND": int(
                    reasons.get(
                        "action_bound",
                        0,
                    )
                ),
                "DOMAIN_CONSTRAINT": int(
                    reasons.get(
                        "domain_constraint",
                        0,
                    )
                ),
                "LYAPUNOV_DECREASE": int(
                    reasons.get(
                        "lyapunov_decrease",
                        0,
                    )
                ),
                "Strict Delta-V < 0": int(
                    item.get(
                        "strict_decrease_count",
                        0,
                    )
                ),
            }
        )

    return rows


def proposal_claim_rows(
    claims: dict[str, Any],
) -> list[dict[str, str]]:
    source = claims.get("proposal_claims", [])

    if not isinstance(
        source,
        list,
    ):
        raise TypeError("Proposal claims must be list")

    rows: list[dict[str, str]] = []

    for item in source:
        if not isinstance(
            item,
            dict,
        ):
            continue

        rows.append(
            {
                "ID": str(item["claim_id"]),
                "Status": str(item["status"]),
                "Statement": str(item["statement"]),
                "Limitation": str(item.get("limitation", "")),
            }
        )

    return rows


def unsupported_claim_rows(
    claims: dict[str, Any],
) -> list[dict[str, str]]:
    source = claims.get("explicitly_unsupported", [])

    if not isinstance(
        source,
        list,
    ):
        raise TypeError("Unsupported claims must be list")

    rows: list[dict[str, str]] = []

    for item in source:
        if not isinstance(
            item,
            dict,
        ):
            continue

        rows.append(
            {
                "ID": str(item["claim_id"]),
                "Status": str(item["status"]),
                "Statement": str(item["statement"]),
            }
        )

    return rows


def figure_paths(
    index: dict[str, Any],
) -> list[tuple[str, Path]]:
    figures = index.get("figures", [])

    if not isinstance(
        figures,
        list,
    ):
        raise TypeError("Figure index must contain list")

    output: list[tuple[str, Path]] = []

    for item in figures:
        if not isinstance(
            item,
            dict,
        ):
            continue

        path = ROOT / str(item["path"])

        output.append(
            (
                str(item["title"]),
                path,
            )
        )

    return output


def render() -> None:
    evidence = load_json(EVIDENCE_PATH)

    claims = load_json(CLAIM_MATRIX_PATH)

    figure_index = load_json(FIGURE_INDEX_PATH)

    st.set_page_config(
        page_title=("Q-VLA Forge | Safety Evidence"),
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ Sprint 5 Safety Evidence")

    st.caption(
        "Frozen Phase 1 safety and robustness "
        "evidence for autonomous driving and robotics."
    )

    metadata = evidence["metadata"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Principal Seeds",
        len(metadata["principal_seeds"]),
    )

    col2.metric(
        "Domains",
        len(metadata["domains"]),
    )

    col3.metric(
        "Safety Methods",
        len(metadata["methods"]),
    )

    col4.metric(
        "Scientific Scope",
        "FROZEN",
    )

    st.info(
        "This page displays frozen Sprint 5 "
        "evidence only. It performs no training, "
        "simulation, policy execution, retuning, "
        "or raw scientific recomputation."
    )

    st.header("Clean Safety")

    driving_none = mean_value(
        evidence,
        "autonomous_driving",
        "none",
        "violation_step_rate",
    )

    driving_clip = mean_value(
        evidence,
        "autonomous_driving",
        "clipping",
        "violation_step_rate",
    )

    driving_lyap = mean_value(
        evidence,
        "autonomous_driving",
        "lyapunov",
        "violation_step_rate",
    )

    robotics_none = mean_value(
        evidence,
        "robotics",
        "none",
        "violation_step_rate",
    )

    robotics_clip = mean_value(
        evidence,
        "robotics",
        "clipping",
        "violation_step_rate",
    )

    robotics_lyap = mean_value(
        evidence,
        "robotics",
        "lyapunov",
        "violation_step_rate",
    )

    a, b = st.columns(2)

    with a:
        st.subheader("Autonomous Driving")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "NONE violation rate",
            fmt(driving_none),
        )

        c2.metric(
            "CLIPPING",
            fmt(driving_clip),
        )

        c3.metric(
            "LYAPUNOV",
            fmt(driving_lyap),
        )

    with b:
        st.subheader("Robotics")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "NONE violation rate",
            fmt(robotics_none),
        )

        c2.metric(
            "CLIPPING",
            fmt(robotics_clip),
        )

        c3.metric(
            "LYAPUNOV",
            fmt(robotics_lyap),
        )

    st.success(
        "Under the frozen synthetic proxy safety "
        "contracts, both explicit safety filters "
        "reduced measured clean executed "
        "violation-step events to zero in both "
        "evaluated domains."
    )

    st.warning(
        "This is empirical proxy evidence, not a "
        "formal safety guarantee or certification result."
    )

    st.dataframe(
        clean_rows(evidence),
        use_container_width=True,
        hide_index=True,
    )

    st.header("Robustness Evidence")

    robustness = evidence["gaussian_robustness"]

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Gaussian levels",
        len(robustness["perturbed_sigmas"]),
    )

    state_cells = evidence["structured_state_robustness"]["cells"]

    action_cells = evidence["action_robustness"]["cells"]

    r2.metric(
        "Structured-state cells",
        len(state_cells),
    )

    r3.metric(
        "Structured-action cells",
        len(action_cells),
    )

    st.caption(
        "Gaussian and structured-state studies "
        "perturb the policy observation while the "
        "explicit safety layer retains true simulator state."
    )

    st.header("Unsafe Action Recovery")

    global_action = evidence["action_robustness"]["global"]

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "Unsafe perturbed steps",
        int(global_action["unsafe_perturbed_steps"]),
    )

    a2.metric(
        "Recovered",
        int(global_action["recovered_unsafe_steps"]),
    )

    a3.metric(
        "Unresolved",
        int(global_action["unresolved_unsafe_steps"]),
    )

    a4.metric(
        "Recovery fraction",
        fmt(float(global_action["recovery_fraction"])),
    )

    st.dataframe(
        action_rows(evidence),
        use_container_width=True,
        hide_index=True,
    )

    environment = evidence["action_robustness"]["environment_interface"]

    st.caption(
        "Environment-interface adjustments: "
        f"{environment['environment_interface_adjustments_count']}. "
        "These adjustments are kept separate from "
        "explicit safety-filter interventions."
    )

    st.header("Lyapunov Mechanisms")

    st.dataframe(
        mechanism_rows(evidence),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "LYAPUNOV_DECREASE intervention reason "
        "and strict empirical Delta-V < 0 are "
        "distinct quantities."
    )

    st.header("Cross-Domain Reuse")

    architecture = evidence["cross_domain"]["architecture_reuse"]

    x1, x2, x3 = st.columns(3)

    x1.metric(
        "Shared safety interface",
        str(bool(architecture["shared_safety_interface"])),
    )

    x2.metric(
        "Same policy weights",
        str(bool(architecture["same_policy_weights"])),
    )

    x3.metric(
        "Universal controller",
        str(bool(architecture["universal_controller_supported"])),
    )

    st.caption(
        "Cross-domain reuse applies to the "
        "framework and evaluation infrastructure; "
        "constraints, predictors, Lyapunov semantics, "
        "and policy weights remain domain-specific."
    )

    st.header("Proposal Figures")

    figures = figure_paths(figure_index)

    for i in range(
        0,
        len(figures),
        2,
    ):
        columns = st.columns(2)

        for offset in range(2):
            index = i + offset

            if index >= len(figures):
                continue

            title, path = figures[index]

            with columns[offset]:
                st.subheader(title)

                if path.is_file():
                    st.image(
                        str(path),
                        use_container_width=True,
                    )
                else:
                    st.error("Figure missing: " f"{path}")
    st.header("Proposal Claim Matrix")

    proposal_claims = proposal_claim_rows(claims)

    supported = [row for row in proposal_claims if row["Status"] == "SUPPORTED"]

    limited = [
        row for row in proposal_claims if row["Status"] == "SUPPORTED_WITH_LIMITATION"
    ]

    unsupported = unsupported_claim_rows(claims)

    with st.expander(
        "SUPPORTED",
        expanded=True,
    ):
        st.dataframe(
            supported,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(
        "SUPPORTED WITH LIMITATION",
        expanded=True,
    ):
        st.dataframe(
            limited,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(
        "NOT SUPPORTED",
        expanded=False,
    ):
        st.dataframe(
            unsupported,
            use_container_width=True,
            hide_index=True,
        )

    st.header("Limitations")

    st.warning(
        "Synthetic proxy environments | "
        "Three principal policy seeds | "
        "Privileged true-state safety access | "
        "No physical vehicle/robot | "
        "No formal stability proof | "
        "No certification"
    )

    limitation_payload = evidence.get(
        "limitations",
        {},
    )

    if isinstance(
        limitation_payload,
        dict,
    ):
        limitation_items = limitation_payload.get(
            "limitations",
            [],
        )
    else:
        limitation_items = []

    with st.expander("Full Frozen Limitation List"):
        for item in limitation_items:
            if isinstance(
                item,
                dict,
            ):
                st.write(
                    "- "
                    + str(
                        item.get(
                            "statement",
                            "",
                        )
                    )
                )

    st.header("Evidence Artifacts")

    evidence_paths = {
        "JSON": EVIDENCE_PATH,
        "CSV": CSV_PATH,
        "Markdown": MARKDOWN_PATH,
        "Claim Matrix": CLAIM_MATRIX_PATH,
        "Manifest": MANIFEST_PATH,
        "Figure Index": FIGURE_INDEX_PATH,
    }

    for label, artifact_path in evidence_paths.items():
        status = "READY" if artifact_path.is_file() else "MISSING"

        st.code(f"{label}: " f"{artifact_path.relative_to(ROOT)} " f"[{status}]")

    st.caption(
        "Read-only proposal evidence. "
        "The dashboard performs no scientific recomputation."
    )


if __name__ == "__main__":
    render()
