"""Final Phase 1 evidence dashboard for Q-VLA Forge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from q_vla_forge.evaluation.dashboard_evidence import (
    DOMAINS,
    MISSING_CANONICAL_EVIDENCE,
    REQUIRED_SEEDS,
    MissingCanonicalEvidenceError,
    load_dashboard_evidence,
)

ROOT = Path(__file__).resolve().parents[1]

DOMAIN_OPTIONS = {
    "All Domains": None,
    "Autonomous Driving": "Driving",
    "Robotics": "Robotics",
}

DOMAIN_METADATA = {
    "All Domains": None,
    "Autonomous Driving": "autonomous_driving",
    "Robotics": "robotics",
}


st.set_page_config(
    page_title="Q-VLA Forge | Phase 1 Evidence",
    page_icon="⚛️",
    layout="wide",
)


@st.cache_data
def load_evidence() -> dict[str, Any]:
    """Load the canonical frozen Sprint 7 evidence package."""
    return load_dashboard_evidence(ROOT)


def filter_domain_rows(
    rows: list[dict[str, str]],
    selected_domain: str,
) -> list[dict[str, str]]:
    """Filter canonical table rows without recomputing statistics."""
    domain_label = DOMAIN_OPTIONS[selected_domain]

    if domain_label is None:
        return rows

    if not rows:
        return rows

    if "Domain" not in rows[0]:
        return rows

    return [row for row in rows if row.get("Domain") == domain_label]


def render_header(
    evidence: dict[str, Any],
) -> None:
    """Render Phase 1 dashboard headline status."""
    st.title("Q-VLA Forge — Phase 1 Evidence Dashboard")

    st.caption(
        "Frozen reviewer-facing evidence for the "
        "autonomous-driving and robotics proxy domains."
    )

    (
        col1,
        col2,
        col3,
        col4,
        col5,
        col6,
    ) = st.columns(6)

    col1.metric(
        "Domains",
        len(evidence["domains"]),
    )

    col2.metric(
        "Seeds",
        len(evidence["seeds"]),
    )

    col3.metric(
        "Challenge Bottlenecks",
        len(evidence["challenge_bottlenecks"]),
    )

    col4.metric(
        "Final Figures",
        len(evidence["figures"]),
    )

    col5.metric(
        "Final Tables",
        len(evidence["tables"]),
    )

    col6.metric(
        "Direct Factorial Runs",
        evidence["full_system_status"]["DIRECT"],
    )

    st.success("Evidence Status: PHASE 1 FROZEN")

    (
        control1,
        control2,
        control3,
        control4,
    ) = st.columns(4)

    control1.metric(
        "New Training",
        "No",
    )

    control2.metric(
        "New Experiments",
        "No",
    )

    control3.metric(
        "Quantum Hardware Used",
        "No",
    )

    control4.metric(
        "Production Validation",
        "No",
    )

    st.caption(
        "Primary statistics: mean ± sample SD, "
        "n = 3 locked seeds "
        f"({', '.join(str(seed) for seed in REQUIRED_SEEDS)})."
    )


def render_domain_selector() -> str:
    """Render domain-level presentation filter."""
    st.sidebar.header("Phase 1 Evidence")

    selected_domain = st.sidebar.selectbox(
        "Domain",
        options=list(DOMAIN_OPTIONS),
        index=0,
    )

    st.sidebar.caption(
        "Statistics are read from frozen evidence. "
        "Changing this selector does not recompute results."
    )

    st.sidebar.markdown("### Locked Seeds")

    for seed in REQUIRED_SEEDS:
        st.sidebar.code(str(seed))

    return selected_domain


def render_compression_card(
    evidence: dict[str, Any],
    selected_domain: str,
) -> None:
    """Render criterion-specific compression evidence."""
    st.subheader("Compression")

    st.caption(
        "Frozen criterion: ≥2.0× compression and " "≤5% relative MSE degradation."
    )

    outcomes = evidence["criterion_outcomes"]["compression"]

    selected_label = DOMAIN_OPTIONS[selected_domain]

    methods = (
        "INT8",
        "SVD",
        "TT/MPS",
    )

    columns = st.columns(3)

    for column, method in zip(
        columns,
        methods,
        strict=True,
    ):
        domain_results = outcomes[method]

        if selected_label is None:
            statuses = set(domain_results.values())

            if statuses == {"PASS"}:
                value = "PASS"
                detail = "both domains"
            elif statuses == {"FAIL"}:
                value = "FAIL"
                detail = "both domains"
            else:
                value = "MIXED"
                detail = "domain dependent"
        else:
            value = domain_results[selected_label]

            detail = selected_label

        column.metric(
            method,
            value,
        )

        column.caption(detail)

    st.info(
        "INT8 is the only evaluated compression "
        "method that satisfies the frozen joint "
        "criterion in both proxy domains."
    )


def render_training_card(
    evidence: dict[str, Any],
) -> None:
    """Render the frozen training-efficiency conclusion."""
    st.subheader("Training Efficiency")

    result = evidence["criterion_outcomes"]["training_efficiency"]

    st.metric(
        "Robust ≥10% Efficiency Improvement",
        result["phase1_result"],
    )

    st.caption("Frozen target: " + result["threshold"])

    st.warning(
        "Isolated target reaches are not interpreted "
        "as a robust cross-domain efficiency improvement."
    )


def _domain_rl_rows(
    evidence: dict[str, Any],
    selected_domain: str,
) -> list[dict[str, str]]:
    rows = evidence["tables"]["rl"]

    return filter_domain_rows(
        rows,
        selected_domain,
    )


def _sum_target_reaches(
    rows: list[dict[str, str]],
    policy: str,
) -> tuple[int, int]:
    reached = 0
    total = 0

    for row in rows:
        if row["Policy"] != policy:
            continue

        left, right = row["Target Reaches"].split("/")

        reached += int(left)

        total += int(right)

    return (
        reached,
        total,
    )


def render_rl_card(
    evidence: dict[str, Any],
    selected_domain: str,
) -> None:
    """Render RL target attainment and separate PQC compactness."""
    st.subheader("RL / QML")

    rows = _domain_rl_rows(
        evidence,
        selected_domain,
    )

    policies = (
        (
            "Classical PPO / MLP",
            "Classical PPO",
        ),
        (
            "Matched classical control",
            "Matched Classical",
        ),
        (
            "QML / PQC",
            "QML / PQC",
        ),
    )

    columns = st.columns(3)

    for column, (
        policy,
        label,
    ) in zip(
        columns,
        policies,
        strict=True,
    ):
        reached, total = _sum_target_reaches(
            rows,
            policy,
        )

        column.metric(
            label,
            f"{reached}/{total}",
        )

    st.markdown("#### PQC Actor Compactness")

    compactness = evidence["criterion_outcomes"]["rl_alignment"]["pqc_compactness"]

    selected_label = DOMAIN_OPTIONS[selected_domain]

    if selected_label is None:
        left, right = st.columns(2)

        left.metric(
            "Driving Parameter Reduction",
            compactness["Driving"],
        )

        right.metric(
            "Robotics Parameter Reduction",
            compactness["Robotics"],
        )
    else:
        st.metric(
            f"{selected_label} Parameter Reduction",
            compactness[selected_label],
        )

    st.warning(
        "Actor compactness is separate from "
        "sample efficiency and policy performance. "
        "Phase 1 did not demonstrate a QML "
        "sample-efficiency advantage."
    )


def render_safety_card(
    evidence: dict[str, Any],
    selected_domain: str,
) -> None:
    """Render clean empirical safety evidence."""
    st.subheader("Safety")

    safety = evidence["criterion_outcomes"]["safety"]

    selected_label = DOMAIN_OPTIONS[selected_domain]

    domains = (
        (
            "Driving",
            "Autonomous Driving",
        ),
        (
            "Robotics",
            "Robotics",
        ),
    )

    if selected_label is not None:
        domains = tuple(item for item in domains if item[0] == selected_label)

    for domain_key, domain_label in domains:
        st.markdown(f"**{domain_label}**")

        columns = st.columns(3)

        for column, method in zip(
            columns,
            (
                "NONE",
                "CLIPPING",
                "LYAPUNOV",
            ),
            strict=True,
        ):
            column.metric(
                method,
                safety[domain_key][method],
            )

    st.warning(
        "Empirical proxy result only. Zero observed "
        "violations do not establish formal stability, "
        "certification, or production-safety guarantees."
    )


def render_bottleneck_overview(
    evidence: dict[str, Any],
    selected_domain: str,
) -> None:
    """Render the four challenge bottleneck sections."""
    st.header("Challenge Bottleneck Overview")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            render_compression_card(
                evidence,
                selected_domain,
            )

        with st.container(border=True):
            render_rl_card(
                evidence,
                selected_domain,
            )

    with right:
        with st.container(border=True):
            render_training_card(evidence)

        with st.container(border=True):
            render_safety_card(
                evidence,
                selected_domain,
            )


def render_final_figures(
    evidence: dict[str, Any],
    selected_domain: str,
) -> None:
    """Render the seven frozen Sprint 7.7 figures."""
    st.header("Final Phase 1 Figures")

    st.info(
        "These are frozen Sprint 7.7 figures. "
        "The domain selector does not regenerate "
        "or alter the underlying plots."
    )

    domain_metadata = DOMAIN_METADATA[selected_domain]

    figures = []

    for figure in evidence["figures"]:
        domains = figure.get(
            "domains",
            [],
        )

        if domain_metadata is not None and domain_metadata not in domains:
            continue

        figures.append(figure)

    for index in range(
        0,
        len(figures),
        2,
    ):
        columns = st.columns(2)

        for offset in range(2):
            position = index + offset

            if position >= len(figures):
                continue

            figure = figures[position]

            path = ROOT / str(figure["output_path"]).replace(
                "\\",
                "/",
            )

            with columns[offset]:
                st.subheader(figure["title"])

                st.image(
                    str(path),
                    use_container_width=True,
                )

                st.caption(figure["scientific_scope"])

                with st.expander("Figure provenance"):
                    st.write(
                        "**Figure ID:**",
                        figure["figure_id"],
                    )

                    st.write(
                        "**Statistical protocol:**",
                        figure["statistical_protocol"],
                    )

                    st.write("**Source artifacts:**")

                    for source in figure["source_artifacts"]:
                        st.code(source)

                    st.write("**Limitations:**")

                    for limitation in figure["limitations"]:
                        st.write(f"- {limitation}")


def render_final_tables(
    evidence: dict[str, Any],
    selected_domain: str,
) -> None:
    """Render the five canonical Sprint 7.8 result tables."""
    st.header("Final Result Tables")

    table_manifest = evidence["table_manifest"]

    metadata = {item["table_id"]: item for item in table_manifest["tables"]}

    for table_id in (
        "compression",
        "training",
        "rl",
        "safety",
        "cross_domain",
    ):
        record = metadata[table_id]

        rows = filter_domain_rows(
            evidence["tables"][table_id],
            selected_domain,
        )

        st.subheader(record["title"])

        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )

        with st.expander("Table provenance"):
            st.write(
                "**Statistical protocol:**",
                record["statistical_protocol"],
            )

            st.write("**Source artifacts:**")

            for source in record["source_artifacts"]:
                st.code(source)

            st.write("**Limitations:**")

            for limitation in record["limitations"]:
                st.write(f"- {limitation}")


def _ablation_rows(
    evidence: dict[str, Any],
    area: str,
) -> list[dict[str, str]]:
    return [row for row in evidence["component_ablation"] if row["area"] == area]


def render_ablation(
    evidence: dict[str, Any],
) -> None:
    """Render measured component evidence and full-system boundary."""
    st.header("Ablation Evidence")

    st.subheader("Compression Ablation")

    st.dataframe(
        pd.DataFrame(
            _ablation_rows(
                evidence,
                "compression",
            )
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("QML Ablation")

    st.dataframe(
        pd.DataFrame(
            _ablation_rows(
                evidence,
                "rl_qml",
            )
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Safety Ablation")

    st.dataframe(
        pd.DataFrame(
            _ablation_rows(
                evidence,
                "safety",
            )
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Full-System Ablation")

    status = evidence["full_system_status"]

    (
        col1,
        col2,
        col3,
    ) = st.columns(3)

    col1.metric(
        "DIRECT",
        status["DIRECT"],
    )

    col2.metric(
        "COMPONENT_ONLY",
        status["COMPONENT_ONLY"],
    )

    col3.metric(
        "NOT_EVALUATED",
        status["NOT_EVALUATED"],
    )

    st.warning(
        "No matched integrated Compression × QML × "
        "Safety factorial configuration was directly "
        "executed in Phase 1. Component results are "
        "not combined into synthetic full-system "
        "performance estimates."
    )

    phase1 = evidence["full_system"]["phase1"]

    matrix = phase1.get(
        "matrix",
        [],
    )

    if (
        isinstance(
            matrix,
            list,
        )
        and matrix
    ):
        st.dataframe(
            pd.DataFrame(matrix),
            width="stretch",
            hide_index=True,
        )

    st.caption(evidence["full_system"]["scientific_conclusion"])


def _claim_status_label(
    status: str,
) -> str:
    mapping = {
        "supported": "SUPPORTED",
        "supported_with_limitation": "LIMITED",
        "not_demonstrated": "NOT DEMONSTRATED",
        "blocked": "BLOCKED",
    }

    return mapping.get(
        status.lower(),
        status.upper(),
    )


def render_claims(
    evidence: dict[str, Any],
) -> None:
    """Render final proposal claims and blocked claim controls."""
    st.header("Proposal Claim Evidence")

    rows = []

    for claim in evidence["claim_registry"]:
        rows.append(
            {
                "Claim ID": claim["claim_id"],
                "Area": claim["area"],
                "Status": _claim_status_label(claim["status"]),
                "Statement": claim["statement"],
                "Limitation": claim["limitation"],
                "Evidence": claim["evidence"],
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Blocked Claim Controls")

    blocked = [
        {
            "Claim Control": key,
            "Allowed": value,
        }
        for key, value in sorted(evidence["claim_controls"].items())
    ]

    st.dataframe(
        pd.DataFrame(blocked),
        width="stretch",
        hide_index=True,
    )

    st.warning(
        "Quantum advantage, quantum speedup, "
        "formal Lyapunov stability, production readiness, "
        "zero-shot transfer, and full-system superiority "
        "remain blocked by the frozen Phase 1 evidence."
    )


def render_provenance(
    evidence: dict[str, Any],
) -> None:
    """Render canonical evidence provenance."""
    st.header("Evidence Provenance")

    st.info(
        "The final dashboard is a read-only evidence "
        "browser. It performs no training, simulation, "
        "retuning, policy execution, or scientific "
        "recomputation."
    )

    st.subheader("Canonical Sources")

    for label, path in evidence["provenance"].items():
        st.code(f"{label}: {path}")

    st.subheader("Evidence Contract")

    contract_rows = [
        {
            "Property": "Phase",
            "Value": evidence["phase"],
        },
        {
            "Property": "Status",
            "Value": evidence["status"],
        },
        {
            "Property": "Domains",
            "Value": ", ".join(DOMAINS),
        },
        {
            "Property": "Seeds",
            "Value": ", ".join(str(seed) for seed in REQUIRED_SEEDS),
        },
        {
            "Property": "New training",
            "Value": str(evidence["new_training"]),
        },
        {
            "Property": "New experiments",
            "Value": str(evidence["new_experiments"]),
        },
        {
            "Property": "New scientific results",
            "Value": str(evidence["new_scientific_results"]),
        },
        {
            "Property": "Synthetic metric composition",
            "Value": str(evidence["synthetic_metric_composition"]),
        },
    ]

    st.dataframe(
        pd.DataFrame(contract_rows),
        width="stretch",
        hide_index=True,
    )


def render_dashboard(
    evidence: dict[str, Any],
) -> None:
    """Render the final Phase 1 evidence browser."""
    selected_domain = render_domain_selector()

    render_header(evidence)

    tabs = st.tabs(
        [
            "Overview",
            "Final Figures",
            "Final Tables",
            "Ablation",
            "Proposal Claims",
            "Provenance",
        ]
    )

    with tabs[0]:
        render_bottleneck_overview(
            evidence,
            selected_domain,
        )

        st.header("Phase 1 Bottleneck Scorecard")

        st.dataframe(
            pd.DataFrame(evidence["bottleneck_scorecard"]),
            width="stretch",
            hide_index=True,
        )

    with tabs[1]:
        render_final_figures(
            evidence,
            selected_domain,
        )

    with tabs[2]:
        render_final_tables(
            evidence,
            selected_domain,
        )

    with tabs[3]:
        render_ablation(evidence)

    with tabs[4]:
        render_claims(evidence)

    with tabs[5]:
        render_provenance(evidence)


def main() -> None:
    """Run the final Q-VLA Forge Phase 1 dashboard."""
    try:
        evidence = load_evidence()

    except MissingCanonicalEvidenceError as exc:
        st.error(MISSING_CANONICAL_EVIDENCE)

        st.code(str(exc))

        st.warning(
            "Required final-validation evidence is "
            "missing or invalid. The dashboard will "
            "not substitute historical artifacts, "
            "generate zeros, or recompute results."
        )

        st.stop()

    render_dashboard(evidence)


if __name__ == "__main__":
    main()
