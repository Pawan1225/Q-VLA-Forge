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

    column1, column2, column3, column4, column5 = st.columns(5)

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
                f"± {summary['driving']['test_mse']['std']:.6f}"
            ),
        )

        metric_col2.metric(
            "Driving MAE",
            (
                f"{summary['driving']['test_mae']['mean']:.6f} "
                f"± {summary['driving']['test_mae']['std']:.6f}"
            ),
        )

        metric_col3.metric(
            "Robotics MSE",
            (
                f"{summary['robotics']['test_mse']['mean']:.6f} "
                f"± {summary['robotics']['test_mse']['std']:.6f}"
            ),
        )

        metric_col4.metric(
            "Robotics MAE",
            (
                f"{summary['robotics']['test_mae']['mean']:.6f} "
                f"± {summary['robotics']['test_mae']['std']:.6f}"
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

    st.subheader("Sprint Progress")

    progress = pd.DataFrame(
        {
            "Sprint": [
                "0.1 Repository",
                "0.2 Environment",
                "0.3 Classical Stack",
                "0.4 Quantum Stack",
                "0.5 Experiment Foundation",
                "0.6 Configuration",
                "0.7 Dashboard",
                "1 Shared AI/DL Baseline",
                "1.11 Driving Baseline",
                "1.12 Robotics Baseline",
                "1.13 Multi-Seed Validation",
                "1.14 Sprint 1 Final Quality Gate",
                "2 Compression",
                "3 Training Efficiency",
                "4 RL + QML",
                "5 Safety + Robustness",
                "6 Unified Architecture",
                "7 Validation",
            ],
            "Status": [
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Pending",
                "Pending",
                "Pending",
                "Pending",
                "Pending",
                "Pending",
            ],
        }
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
    """Render the planned Q-VLA Forge ablation structure."""
    st.header("Ablation Dashboard")

    st.markdown("""
The final validation sprint will compare:

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

    if dataframe.empty:
        st.info("Ablation results have not " "been generated yet.")
        return

    st.caption("Experiment records will populate " "this section during Sprint 7.")


def render_evidence() -> None:
    """Render proposal evidence tracking."""
    st.header("Proposal Evidence")

    evidence = pd.DataFrame(
        {
            "Claim": [
                "QI compression reduces model size",
                "Compression retains task performance",
                "Compressed models improve efficiency",
                "Hybrid QML policy is trainable",
                "Safety layer reduces violations",
                "Architecture works across both domains",
            ],
            "Evidence": [
                "Pending Sprint 2",
                "Pending Sprint 2",
                "Pending Sprint 3",
                "Pending Sprint 4",
                "Pending Sprint 5",
                "Pending Sprint 6",
            ],
            "Figure": [
                "Compression Pareto",
                "Compression Pareto",
                "Training Convergence",
                "RL Learning Curve",
                "Safety Violations",
                "Cross-Domain Comparison",
            ],
            "Proposal Ready": [
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        }
    )

    st.dataframe(
        evidence,
        width="stretch",
        hide_index=True,
    )


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
