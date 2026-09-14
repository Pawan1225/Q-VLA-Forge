from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.data_loader import (
    load_sprint4_rl_evidence,
)

EVIDENCE_PATH = Path("results/rl/evidence/" "sprint4-rl-evidence.json")

FIGURE_DIR = Path("figures/rl")


st.set_page_config(
    page_title="Sprint 4 - RL + Hybrid QML",
    page_icon="⚛️",
    layout="wide",
)


@st.cache_data
def load_evidence() -> dict:
    """Load packaged Sprint 4 evidence only."""
    return load_sprint4_rl_evidence(EVIDENCE_PATH)


def format_auc(
    mean: float,
    sd: float,
) -> str:
    return f"{mean:.3f} ± {sd:.3f}"


def method_table(
    domain_data: dict,
) -> pd.DataFrame:
    rows = []

    labels = {
        "full_ppo": "Full PPO",
        "matched_classical": ("Matched Classical"),
        "hybrid_qml": "Hybrid QML",
    }

    for method in (
        "full_ppo",
        "matched_classical",
        "hybrid_qml",
    ):
        record = domain_data["methods"][method]

        rows.append(
            {
                "Method": labels[method],
                "Actor Parameters": (record["actor_parameters"]),
                "Target Reach": (
                    f"{record['target_reach_count']}" f"/{record['target_total']}"
                ),
                "Normalized AUC": format_auc(
                    record["normalized_auc_mean"],
                    record["normalized_auc_sd"],
                ),
                "Best Progress": format_auc(
                    record["best_progress_mean"],
                    record["best_progress_sd"],
                ),
                "Final Progress": format_auc(
                    record["final_progress_mean"],
                    record["final_progress_sd"],
                ),
            }
        )

    return pd.DataFrame(rows)


def render_domain_section(
    title: str,
    domain_data: dict,
) -> None:
    st.subheader(title)

    compactness = domain_data["compactness"]

    matched = domain_data["matched_budget_representation"]

    (
        col1,
        col2,
        col3,
    ) = st.columns(3)

    col1.metric(
        "Full Actor Parameters",
        f"{compactness['full_actor_parameters']:,}",
    )

    col2.metric(
        "Compact Actor Parameters",
        f"{compactness['compact_actor_parameters']:,}",
    )

    col3.metric(
        "Actor Reduction",
        (f"{compactness['parameter_reduction_percent']:.2f}%"),
    )

    st.dataframe(
        method_table(domain_data),
        use_container_width=True,
        hide_index=True,
    )

    if matched["direction"] == "hybrid_qml":
        direction_text = (
            "Hybrid QML had the higher mean "
            "normalized AUC in this matched-budget "
            "domain comparison."
        )
    else:
        direction_text = (
            "Matched Classical had the higher mean "
            "normalized AUC in this matched-budget "
            "domain comparison."
        )

    st.info(direction_text)

    st.caption(
        "Matched Classical minus Hybrid QML "
        "normalized-AUC delta: "
        f"{matched['matched_minus_qml_auc']:.6f}"
    )


def render_figure(
    filename: str,
    caption: str,
) -> None:
    path = FIGURE_DIR / filename

    if path.exists():
        st.image(
            str(path),
            caption=caption,
            use_container_width=True,
        )
    else:
        st.warning(f"Figure not found: {path}")


data = load_evidence()

st.title("Sprint 4 - RL + Hybrid QML")

st.caption(
    "Reviewer-facing evidence page generated "
    "from the frozen Sprint 4.14 evidence package."
)

if not data:
    st.error("Sprint 4 evidence package is unavailable.")
    st.stop()


# -------------------------------------------------
# Headline evidence
# -------------------------------------------------

st.header("Headline Evidence")

target_reach = data["target_reach"]

driving = data["domains"]["autonomous_driving"]

robotics = data["domains"]["robotics"]

(
    card1,
    card2,
    card3,
    card4,
    card5,
) = st.columns(5)

card1.metric(
    "PPO Target Reach",
    (f"{target_reach['full_ppo']['reached']}" f"/{target_reach['full_ppo']['total']}"),
)

card2.metric(
    "Matched Target Reach",
    (
        f"{target_reach['matched_classical']['reached']}"
        f"/{target_reach['matched_classical']['total']}"
    ),
)

card3.metric(
    "QML Target Reach",
    (
        f"{target_reach['hybrid_qml']['reached']}"
        f"/{target_reach['hybrid_qml']['total']}"
    ),
)

card4.metric(
    "Driving Actor Reduction",
    (f"{driving['compactness']['parameter_reduction_percent']:.2f}%"),
)

card5.metric(
    "Robotics Actor Reduction",
    (f"{robotics['compactness']['parameter_reduction_percent']:.2f}%"),
)


# -------------------------------------------------
# Domain evidence
# -------------------------------------------------

st.header("Domain Results")

render_domain_section(
    "Autonomous Driving",
    driving,
)

render_domain_section(
    "Robotics",
    robotics,
)


# -------------------------------------------------
# Cross-domain evidence
# -------------------------------------------------

st.header("Cross-Domain Interpretation")

cross_domain = data["cross_domain"]

(
    cross1,
    cross2,
    cross3,
) = st.columns(3)

cross1.metric(
    "Architecture Reuse",
    ("SUPPORTED" if cross_domain["architecture_reuse"] else "NOT SUPPORTED"),
)

cross2.metric(
    "QML Advantage",
    (
        "SUPPORTED"
        if cross_domain["robust_cross_domain_qml_advantage"]
        else "NOT SUPPORTED"
    ),
)

cross3.metric(
    "Representation Direction",
    (
        "CONSISTENT"
        if cross_domain["matched_auc_direction_consistent_across_domains"]
        else "DOMAIN DEPENDENT"
    ),
)

st.caption(
    "Architecture reuse means the same policy "
    "architecture was used across both domains. "
    "It does not imply shared trained weights, "
    "transfer learning, or a universal policy."
)


# -------------------------------------------------
# Selected figures
# -------------------------------------------------

st.header("Selected Evidence Figures")

figure_col1, figure_col2 = st.columns(2)

with figure_col1:
    render_figure(
        "driving-matched-ablation.png",
        ("Driving matched-budget " "classical vs Hybrid QML"),
    )

with figure_col2:
    render_figure(
        "robotics-matched-ablation.png",
        ("Robotics matched-budget " "classical vs Hybrid QML"),
    )

figure_col3, figure_col4 = st.columns(2)

with figure_col3:
    render_figure(
        "cross-domain-matched-delta.png",
        ("Cross-domain matched-minus-QML " "normalized AUC"),
    )

with figure_col4:
    render_figure(
        "cross-domain-parameter-compactness.png",
        ("Cross-domain actor-parameter " "compactness"),
    )


# -------------------------------------------------
# Claim matrix
# -------------------------------------------------

st.header("Claim Status")

claim_matrix = data["claim_matrix"]

claim_rows = []

for claim in claim_matrix["claims"]:
    claim_rows.append(
        {
            "Claim": claim["claim_id"],
            "Status": claim["status"],
            "Proposal-Safe Wording": claim["proposal_safe_wording"],
        }
    )

st.dataframe(
    pd.DataFrame(claim_rows),
    use_container_width=True,
    hide_index=True,
)


# -------------------------------------------------
# Pilot boundary
# -------------------------------------------------

st.header("Pilot Boundary")

st.warning("""
- Synthetic proxy environments
- 3 principal seeds
- 20,000 interaction steps per principal run
- 4-qubit simulated PQC
- PennyLane `default.qubit`
- No quantum hardware
- No quantum speedup claim
- No quantum-hardware advantage claim
- No formal statistical significance claim
- No transfer learning or shared trained weights
- No Sprint 4 safety claim
""")

st.caption("Dashboard source: " "results/rl/evidence/" "sprint4-rl-evidence.json")

st.caption(
    "The dashboard performs presentation only; "
    "it does not recompute Sprint 4 science."
)
