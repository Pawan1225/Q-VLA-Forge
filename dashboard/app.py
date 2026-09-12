from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.data_loader import (
    experiments_dataframe,
    load_result_files,
)

RESULTS_DIR = Path("results")


st.set_page_config(
    page_title="Q-VLA Forge Dashboard",
    page_icon="⚛️",
    layout="wide",
)


@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    """Load experiment results into a DataFrame."""
    records = load_result_files(RESULTS_DIR)
    return experiments_dataframe(records)


def render_overview(dataframe: pd.DataFrame) -> None:
    """Render project-level experiment statistics."""
    st.header("Overview")

    total_experiments = len(dataframe)

    if dataframe.empty:
        driving_experiments = 0
        robotics_experiments = 0
        unique_methods = 0
        seeds = 0
    else:
        driving_experiments = int((dataframe["domain"] == "autonomous_driving").sum())

        robotics_experiments = int((dataframe["domain"] == "robotics").sum())

        unique_methods = int(dataframe["method"].dropna().nunique())

        seeds = int(dataframe["seed"].dropna().nunique())

    (
        column1,
        column2,
        column3,
        column4,
        column5,
    ) = st.columns(5)

    column1.metric(
        "Experiments",
        total_experiments,
    )

    column2.metric(
        "Driving",
        driving_experiments,
    )

    column3.metric(
        "Robotics",
        robotics_experiments,
    )

    column4.metric(
        "Methods",
        unique_methods,
    )

    column5.metric(
        "Seeds",
        seeds,
    )

    st.subheader("Frozen Sprint 1 Baseline")

    summary_path = RESULTS_DIR / "baseline-validation-summary.json"

    manifest_path = RESULTS_DIR / "sprint1-baseline-manifest.json"

    if summary_path.exists() and manifest_path.exists():
        summary = json.loads(
            summary_path.read_text(
                encoding="utf-8",
            )
        )

        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8",
            )
        )

        (
            baseline_col1,
            baseline_col2,
            baseline_col3,
            baseline_col4,
        ) = st.columns(4)

        baseline_col1.metric(
            "Baseline",
            manifest["baseline_name"],
        )

        baseline_col2.metric(
            "Version",
            manifest["baseline_version"],
        )

        baseline_col3.metric(
            "Parameters",
            f"{manifest['parameters']:,}",
        )

        baseline_col4.metric(
            "FP32 Size",
            (f"{manifest['fp32_model_size_bytes'] / (1024**2):.4f} MB"),
        )

        (
            metric_col1,
            metric_col2,
            metric_col3,
            metric_col4,
        ) = st.columns(4)

        metric_col1.metric(
            "Driving MSE",
            (
                f"{summary['driving']['test_mse']['mean']:.6f} "
                f"± "
                f"{summary['driving']['test_mse']['std']:.6f}"
            ),
        )

        metric_col2.metric(
            "Driving MAE",
            (
                f"{summary['driving']['test_mae']['mean']:.6f} "
                f"± "
                f"{summary['driving']['test_mae']['std']:.6f}"
            ),
        )

        metric_col3.metric(
            "Robotics MSE",
            (
                f"{summary['robotics']['test_mse']['mean']:.6f} "
                f"± "
                f"{summary['robotics']['test_mse']['std']:.6f}"
            ),
        )

        metric_col4.metric(
            "Robotics MAE",
            (
                f"{summary['robotics']['test_mae']['mean']:.6f} "
                f"± "
                f"{summary['robotics']['test_mae']['std']:.6f}"
            ),
        )

        st.caption(
            "Seeds: "
            + ", ".join(str(seed) for seed in manifest["seeds"])
            + " | Shared latent: "
            + str(manifest["latent_dim"])
            + "-D"
            + " | Action output: "
            + str(manifest["action_dim"])
            + "-D"
        )

    else:
        st.info("Frozen Sprint 1 baseline artifacts " "are not available yet.")

    st.subheader("Frozen Sprint 2 Compression Result")

    (
        sprint2_col1,
        sprint2_col2,
        sprint2_col3,
        sprint2_col4,
    ) = st.columns(4)

    sprint2_col1.metric(
        "Validation Results",
        "18",
    )

    sprint2_col2.metric(
        "Ablation Points",
        "20",
    )

    sprint2_col3.metric(
        "Frozen Artifacts",
        "17",
    )

    sprint2_col4.metric(
        "Cross-Domain Pareto",
        "INT8",
    )

    st.caption(
        "INT8 was the only tested compression "
        "family that satisfied the ≥2× compression "
        "and ≤5% relative test-MSE-change pilot "
        "criterion across all three seeds in both "
        "domains."
    )

    st.caption(
        "TT/open-boundary MPS via TT-SVD was "
        "evaluated as one quantum-inspired "
        "tensor-network family. No quantum "
        "hardware or native compressed-runtime "
        "speedup is claimed."
    )

    sprint2_manifest_path = (
        RESULTS_DIR / "compression" / "sprint2-compression-manifest.json"
    )

    if sprint2_manifest_path.exists():
        sprint2_manifest = json.loads(
            sprint2_manifest_path.read_text(
                encoding="utf-8",
            )
        )

        st.caption(
            "Sprint 2 manifest status: "
            f"{sprint2_manifest['status']} "
            "| Validation seeds: "
            + ", ".join(str(seed) for seed in sprint2_manifest["validation_seeds"])
            + " | Quantum hardware used: "
            + str(sprint2_manifest["quantum_hardware_used"])
        )

    st.subheader("Sprint Progress")

    progress = pd.DataFrame(
        [
            (
                "0.1 Repository",
                "Complete",
            ),
            (
                "0.2 Environment",
                "Complete",
            ),
            (
                "0.3 Classical Stack",
                "Complete",
            ),
            (
                "0.4 Quantum Stack",
                "Complete",
            ),
            (
                "0.5 Experiment Foundation",
                "Complete",
            ),
            (
                "0.6 Configuration",
                "Complete",
            ),
            (
                "0.7 Dashboard",
                "Complete",
            ),
            (
                "1 Shared AI/DL Baseline",
                "Complete",
            ),
            (
                "1.11 Driving Baseline",
                "Complete",
            ),
            (
                "1.12 Robotics Baseline",
                "Complete",
            ),
            (
                "1.13 Multi-Seed Validation",
                "Complete",
            ),
            (
                "1.14 Sprint 1 Final Quality Gate",
                "Complete",
            ),
            (
                "2 Compression",
                "Complete",
            ),
            (
                "2.1 Compression Contracts",
                "Complete",
            ),
            (
                "2.2 Compression Target Analysis",
                "Complete",
            ),
            (
                "2.3 INT8 Quantization",
                "Complete",
            ),
            (
                "2.4 SVD Low-Rank Compression",
                "Complete",
            ),
            (
                "2.5 TT / TT-SVD",
                "Complete",
            ),
            (
                "2.6 MPS Mapping",
                "Complete",
            ),
            (
                "2.7 Driving Compression Experiments",
                "Complete",
            ),
            (
                "2.8 Robotics Compression Experiments",
                "Complete",
            ),
            (
                "2.9 Three-Seed Compression Validation",
                "Complete",
            ),
            (
                "2.10 Compression–Accuracy Pareto",
                "Complete",
            ),
            (
                "2.11 Ablation + Classical vs QI",
                "Complete",
            ),
            (
                "2.12 Dashboard + Proposal Evidence",
                "Complete",
            ),
            (
                "2.13 Sprint 2 Final Quality Gate",
                "Complete",
            ),
            (
                "3 Training Efficiency",
                "Pending",
            ),
            (
                "4 RL + QML",
                "Pending",
            ),
            (
                "5 Safety + Robustness",
                "Pending",
            ),
            (
                "6 Unified Architecture",
                "Pending",
            ),
            (
                "7 Validation",
                "Pending",
            ),
        ],
        columns=[
            "Sprint",
            "Status",
        ],
    )

    st.dataframe(
        progress,
        width="stretch",
        hide_index=True,
    )


def render_experiment_tracker(
    dataframe: pd.DataFrame,
) -> None:
    """Render the full experiment tracker."""
    st.header("Experiment Tracker")

    if dataframe.empty:
        st.info("No experiment results are available yet.")
        return

    domains = sorted(dataframe["domain"].dropna().unique().tolist())

    methods = sorted(dataframe["method"].dropna().unique().tolist())

    domain_filter = st.multiselect(
        "Domain",
        options=domains,
        default=domains,
    )

    method_filter = st.multiselect(
        "Method",
        options=methods,
        default=methods,
    )

    filtered = dataframe.copy()

    if domain_filter:
        filtered = filtered[filtered["domain"].isin(domain_filter)]

    if method_filter:
        filtered = filtered[filtered["method"].isin(method_filter)]

    st.dataframe(
        filtered,
        width="stretch",
        hide_index=True,
    )


def render_benchmarks(
    dataframe: pd.DataFrame,
) -> None:
    """Render benchmark metrics from experiment records."""
    st.header("Benchmark Dashboard")

    if dataframe.empty:
        st.info("Benchmark results will appear here " "once experiments are executed.")
        return

    metric_columns = [
        column for column in dataframe.columns if column.startswith("metric_")
    ]

    if not metric_columns:
        st.info("No benchmark metrics are available yet.")
        return

    selected_metric = st.selectbox(
        "Select metric",
        metric_columns,
    )

    chart_data = dataframe[
        [
            "experiment_id",
            selected_metric,
        ]
    ].dropna()

    if chart_data.empty:
        st.info("No values are available for this metric.")
        return

    chart_data = chart_data.set_index("experiment_id")

    st.bar_chart(chart_data)


def render_ablation(
    dataframe: pd.DataFrame,
) -> None:
    """Render current and planned Q-VLA Forge ablations."""
    st.header("Ablation Dashboard")

    st.write(
        "Sprint 2 compression-target ablation is "
        "complete. The final system-level ablation "
        "in Sprint 7 will compare:"
    )

    st.markdown("""
| Variant | Compression | QML | Safety |
|---|---|---|---|
| Baseline | No | No | No |
| Compression Only | Yes | No | No |
| QML Only | No | Yes | No |
| Safety Only | No | No | Yes |
| Compression + QML | Yes | Yes | No |
| Compression + Safety | Yes | No | Yes |
| QML + Safety | No | Yes | Yes |
| Full Hybrid | Yes | Yes | Yes |
""")

    st.subheader("Sprint 2 Compression Ablation")

    st.metric(
        "Completed Target-Level Points",
        "20",
    )

    st.caption(
        "Classical SVD and quantum-inspired "
        "TT/MPS were evaluated on matched "
        "architectural targets using seed 42."
    )

    st.caption(
        "Targets: fusion, latent, action, " "fusion + latent, and all eligible layers."
    )

    if dataframe.empty:
        st.info("Experiment-tracker records are not " "available yet.")
        return

    st.caption(
        "Future system-level experiment records "
        "will populate this section during Sprint 7."
    )


def render_evidence() -> None:
    """Render proposal evidence tracking."""
    st.header("Proposal Evidence")

    evidence = pd.DataFrame(
        [
            {
                "Claim": (
                    "INT8 compression exceeds the " "2× pilot target in both domains"
                ),
                "Evidence": ("Sprint 2 — 3-seed validation"),
                "Figure": "Compression Pareto",
                "Proposal Ready": True,
            },
            {
                "Claim": (
                    "INT8 preserves task performance " "under the Sprint 2 criterion"
                ),
                "Evidence": ("Sprint 2 — 3-seed validation"),
                "Figure": "Compression Pareto",
                "Proposal Ready": True,
            },
            {
                "Claim": (
                    "TT/MPS quantum-inspired "
                    "compression was implemented "
                    "and evaluated"
                ),
                "Evidence": ("Sprint 2 — TT-SVD + MPS mapping"),
                "Figure": ("Compression Pareto / Ablation"),
                "Proposal Ready": True,
            },
            {
                "Claim": (
                    "TT/MPS did not satisfy the "
                    "compression-quality pilot "
                    "criterion"
                ),
                "Evidence": ("Sprint 2 — 3-seed validation"),
                "Figure": "Compression Pareto",
                "Proposal Ready": True,
            },
            {
                "Claim": (
                    "Classical SVD and quantum-inspired "
                    "TT/MPS were compared on matched "
                    "architectural targets"
                ),
                "Evidence": ("Sprint 2 — 20-point ablation"),
                "Figure": "Compression Ablation",
                "Proposal Ready": True,
            },
            {
                "Claim": (
                    "Shared architecture operates " "across driving and robotics"
                ),
                "Evidence": ("Sprint 1 + Sprint 2"),
                "Figure": ("Cross-Domain Comparison"),
                "Proposal Ready": True,
            },
            {
                "Claim": ("Compressed representations " "improve training efficiency"),
                "Evidence": "Pending Sprint 3",
                "Figure": "Training Convergence",
                "Proposal Ready": False,
            },
            {
                "Claim": ("Hybrid QML policy is trainable"),
                "Evidence": "Pending Sprint 4",
                "Figure": "RL Learning Curve",
                "Proposal Ready": False,
            },
            {
                "Claim": ("Safety layer reduces violations"),
                "Evidence": "Pending Sprint 5",
                "Figure": "Safety Violations",
                "Proposal Ready": False,
            },
        ]
    )

    st.dataframe(
        evidence,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Sprint 2 Evidence Summary")

    st.markdown("""
**Proposal-safe Sprint 2 conclusions**

- INT8 achieved approximately **3.846× effective whole-model compression** in both domains.
- INT8 satisfied the **≥2× compression and ≤5% relative test-MSE-change** criterion in **3/3 seeds** for both autonomous driving and robotics.
- INT8 was the only tested compression family on the three-seed Pareto frontier in both domains.
- Classical truncated SVD and quantum-inspired TT/MPS were implemented and evaluated as comparison methods.
- TT/MPS achieved greater compression than the selected SVD configurations, but substantially larger task-error degradation.
- TT/MPS is treated as one **quantum-inspired tensor-network family**, not as an independent TT and MPS benchmark.
- No quantum hardware was used in Sprint 2.
- No native INT8, SVD-factorized, or TT/MPS compressed-runtime speedup is claimed.
""")


def main() -> None:
    """Run the Q-VLA Forge dashboard."""
    st.title("Q-VLA Forge — Experiment Dashboard")

    st.caption(
        "Quantum-Enhanced Vision-Language-Action "
        "Pilot for Autonomous Driving and Robotics"
    )

    dataframe = load_dashboard_data()

    page = st.sidebar.radio(
        "Dashboard",
        [
            "Overview",
            "Experiment Tracker",
            "Benchmarks",
            "Ablation",
            "Proposal Evidence",
        ],
    )

    if page == "Overview":
        render_overview(dataframe)

    elif page == "Experiment Tracker":
        render_experiment_tracker(dataframe)

    elif page == "Benchmarks":
        render_benchmarks(dataframe)

    elif page == "Ablation":
        render_ablation(dataframe)

    elif page == "Proposal Evidence":
        render_evidence()


if __name__ == "__main__":
    main()
