from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

TRAINING_DIR = ROOT / "results" / "training"

EVIDENCE_PATH = TRAINING_DIR / "evidence" / "sprint3-training-evidence.json"

ABLATION_PATH = TRAINING_DIR / "ablation" / "classical-vs-qi-ablation-summary.json"


def load_json(
    path: Path,
) -> dict:
    if not path.exists():
        st.error(f"Missing evidence file: {path}")
        st.stop()

    return json.loads(path.read_text(encoding="utf-8"))


def absolute_figure_path(
    relative: str,
) -> Path:
    return ROOT / relative


st.set_page_config(
    page_title=("Q-VLA Forge — Training Efficiency"),
    layout="wide",
)

st.title("Sprint 3 — Training Efficiency")

st.caption(
    "Classical trainable SVD versus quantum-inspired "
    "TT/MPS structured training across autonomous-driving "
    "and robotics VLA proxy tasks."
)

evidence = load_json(EVIDENCE_PATH)


#
# Headline metrics
#

st.subheader("Validation Protocol")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Independent Seeds",
        len(evidence["seeds"]),
    )

with col2:
    st.metric(
        "Domains",
        len(evidence["domains"]),
    )

with col3:
    st.metric(
        "Efficiency Threshold",
        (f"{evidence['challenge_efficiency_threshold_percent']:.0f}%"),
    )

with col4:
    st.metric(
        "Primary Metric",
        "Optimizer Steps",
    )


#
# Three-seed evidence
#

st.subheader("Three-Seed Validation")

rows = []

for item in evidence["table"]:
    if item["step_reduction_mean_percent"] is None:
        step_text = "N/A"
    else:
        step_text = (
            f"{item['step_reduction_mean_percent']:.2f} "
            f"± {item['step_reduction_std_percent']:.2f}%"
        )

    rows.append(
        {
            "Domain": (item["domain"]),
            "Method": (item["method"]),
            "Target Reach": (item["target_reach"]),
            "Parameter Reduction": (f"{item['parameter_reduction_percent']:.2f}%"),
            "Step Reduction": (step_text),
            "Test MSE": (
                f"{item['test_mse_mean']:.6f} " f"± {item['test_mse_std']:.6f}"
            ),
            "Test MAE": (
                f"{item['test_mae_mean']:.6f} " f"± {item['test_mae_std']:.6f}"
            ),
            "Robust ≥10%": ("Yes" if item["robust_ten_percent_efficiency"] else "No"),
        }
    )

st.dataframe(
    pd.DataFrame(rows),
    use_container_width=True,
    hide_index=True,
)


#
# Convergence
#

st.subheader("Convergence")

left, right = st.columns(2)

with left:
    st.markdown("### Autonomous Driving")

    figure = absolute_figure_path(evidence["figures"]["driving_convergence"])

    if figure.exists():
        st.image(
            str(figure),
            use_container_width=True,
        )
    else:
        st.warning(f"Missing figure: {figure}")

with right:
    st.markdown("### Robotics")

    figure = absolute_figure_path(evidence["figures"]["robotics_convergence"])

    if figure.exists():
        st.image(
            str(figure),
            use_container_width=True,
        )
    else:
        st.warning(f"Missing figure: {figure}")


#
# Parameter-vs-efficiency
#

st.subheader("Parameter Reduction vs Training Efficiency")

figure = absolute_figure_path(evidence["figures"]["parameter_vs_efficiency"])

if figure.exists():
    st.image(
        str(figure),
        use_container_width=True,
    )
else:
    st.warning(f"Missing figure: {figure}")


#
# Cross-domain
#

st.subheader("Cross-Domain Comparison")

left, right = st.columns(2)

with left:
    figure = absolute_figure_path(evidence["figures"]["cross_domain_efficiency"])

    if figure.exists():
        st.image(
            str(figure),
            use_container_width=True,
        )

with right:
    figure = absolute_figure_path(
        evidence["figures"]["cross_domain_parameter_reduction"]
    )

    if figure.exists():
        st.image(
            str(figure),
            use_container_width=True,
        )

tt_cross = evidence["cross_domain"]

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "TT/MPS Target Consistency",
        tt_cross["tt_mps_target_consistency"],
    )

with c2:
    st.metric(
        "TT/MPS Efficiency Consistency",
        tt_cross["tt_mps_efficiency_consistency"],
    )

with c3:
    st.metric(
        "Robust Cross-Domain ≥10%",
        ("Yes" if tt_cross["tt_mps_robust_cross_domain_efficiency"] else "No"),
    )


#
# Matched classical-vs-QI ablation
#

st.subheader("Matched Classical vs Quantum-Inspired Ablation")

load_json(ABLATION_PATH)

ablation_rows = []

for item in evidence["matched_ablation"]:
    ablation_rows.append(
        {
            "Domain": (item["domain"]),
            "Seed": (item["seed"]),
            "SVD": (item["svd_configuration"]),
            "TT/MPS": (item["tt_mps_configuration"]),
            "Parameter Difference": (f"{item['parameter_difference_percent']:.2f}%"),
            "SVD Target": ("Reached" if item["svd_reached_target"] else "Not Reached"),
            "TT/MPS Target": (
                "Reached" if item["tt_mps_reached_target"] else "Not Reached"
            ),
            "Interpretation": (item["interpretation"]),
        }
    )

st.dataframe(
    pd.DataFrame(ablation_rows),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Matched-pair selection used parameter counts only; "
    "validation and test performance were not used "
    "to choose the pair."
)


#
# Quantum-inspired method
#

st.subheader("Quantum-Inspired Component")

st.write(evidence["quantum_inspired_method"])

st.write(
    "Quantum hardware used:",
    evidence["quantum_hardware_used"],
)


#
# Supported claims
#

st.subheader("Proposal-Supported Findings")

for claim in evidence["supported_claims"]:
    st.markdown(f"- {claim['statement']}")


#
# Unsupported claims
#

st.subheader("Claims Not Supported by This Sprint")

for claim in evidence["unsupported_claims"]:
    st.markdown(f"- {claim['statement']}")


#
# Architecture interpretation
#

st.subheader("Cross-Domain Architecture Interpretation")

if tt_cross["shared_architecture_claim"]:
    st.success(
        "Supported: a common structured-training framework "
        "was reused across driving and robotics."
    )

if not tt_cross["universal_trained_model_claim"]:
    st.info("Not claimed: one universally trained model " "solving both domains.")


#
# Limitations
#

st.subheader("Limitations")

for limitation in evidence["limitations"]:
    st.markdown(f"- {limitation}")
