from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]

SUMMARY_PATH = ROOT_DIR / "results" / "baseline-validation-summary.json"


def format_mean_std(
    metric: dict[str, Any],
    digits: int = 6,
) -> str:
    """Format a metric as mean ± sample standard deviation."""
    return f"{metric['mean']:.{digits}f} " f"± " f"{metric['std']:.{digits}f}"


st.set_page_config(
    page_title=("Q-VLA Forge | Baseline Validation"),
    layout="wide",
)

st.title("Q-VLA Forge — Baseline Validation")

st.caption("Three independent runs using seeds " "42, 123, and 456.")

if not SUMMARY_PATH.exists():
    st.warning(
        "Baseline validation summary not found. "
        "Run experiments/run_multiseed_baselines.py first."
    )

    st.stop()


payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

driving = payload["driving"]
robotics = payload["robotics"]

st.subheader("Autonomous Driving")

d1, d2, d3 = st.columns(3)

d1.metric(
    "Test MSE",
    format_mean_std(driving["test_mse"]),
)

d2.metric(
    "Test MAE",
    format_mean_std(driving["test_mae"]),
)

d3.metric(
    "Mean Latency",
    (
        f"{driving['mean_latency_ms']['mean']:.3f} "
        f"± "
        f"{driving['mean_latency_ms']['std']:.3f} ms"
    ),
)

st.subheader("Robotics")

r1, r2, r3 = st.columns(3)

r1.metric(
    "Test MSE",
    format_mean_std(robotics["test_mse"]),
)

r2.metric(
    "Test MAE",
    format_mean_std(robotics["test_mae"]),
)

r3.metric(
    "Mean Latency",
    (
        f"{robotics['mean_latency_ms']['mean']:.3f} "
        f"± "
        f"{robotics['mean_latency_ms']['std']:.3f} ms"
    ),
)

st.subheader("Cross-Domain Summary")

summary_table = pd.DataFrame(
    [
        {
            "Domain": "Autonomous Driving",
            "Test MSE Mean": driving["test_mse"]["mean"],
            "Test MSE SD": driving["test_mse"]["std"],
            "Test MAE Mean": driving["test_mae"]["mean"],
            "Test MAE SD": driving["test_mae"]["std"],
            "Mean Latency (ms)": driving["mean_latency_ms"]["mean"],
            "Parameters": driving["parameters"],
            "FP32 Size (MB)": (driving["fp32_model_size_bytes"] / (1024**2)),
        },
        {
            "Domain": "Robotics",
            "Test MSE Mean": robotics["test_mse"]["mean"],
            "Test MSE SD": robotics["test_mse"]["std"],
            "Test MAE Mean": robotics["test_mae"]["mean"],
            "Test MAE SD": robotics["test_mae"]["std"],
            "Mean Latency (ms)": robotics["mean_latency_ms"]["mean"],
            "Parameters": robotics["parameters"],
            "FP32 Size (MB)": (robotics["fp32_model_size_bytes"] / (1024**2)),
        },
    ]
)

st.dataframe(
    summary_table,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Individual Seed Results")

seed_rows: list[dict[str, Any]] = []

for index, seed in enumerate(payload["seeds"]):
    seed_rows.append(
        {
            "Domain": ("Autonomous Driving"),
            "Seed": seed,
            "Test MSE": driving["test_mse"]["values"][index],
            "Test MAE": driving["test_mae"]["values"][index],
            "Mean Latency (ms)": driving["mean_latency_ms"]["values"][index],
        }
    )

    seed_rows.append(
        {
            "Domain": "Robotics",
            "Seed": seed,
            "Test MSE": robotics["test_mse"]["values"][index],
            "Test MAE": robotics["test_mae"]["values"][index],
            "Mean Latency (ms)": robotics["mean_latency_ms"]["values"][index],
        }
    )

st.dataframe(
    pd.DataFrame(seed_rows),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Action-Level Error")

action_rows: list[dict[str, Any]] = []

for metric, result in driving["action_mae"].items():
    action_rows.append(
        {
            "Domain": ("Autonomous Driving"),
            "Action Metric": metric,
            "Mean MAE": result["mean"],
            "SD": result["std"],
        }
    )

for metric, result in robotics["action_mae"].items():
    action_rows.append(
        {
            "Domain": "Robotics",
            "Action Metric": metric,
            "Mean MAE": result["mean"],
            "SD": result["std"],
        }
    )

st.dataframe(
    pd.DataFrame(action_rows),
    use_container_width=True,
    hide_index=True,
)
