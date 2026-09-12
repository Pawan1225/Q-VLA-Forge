from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

COMPRESSION_DIR = ROOT / "results" / "compression"

EVIDENCE_PATH = COMPRESSION_DIR / "evidence" / "sprint2-compression-evidence.json"

ABLATION_PATH = COMPRESSION_DIR / "ablation" / "compression-ablation-seed-42.json"

DRIVING_FIGURE = ROOT / "figures" / "driving_compression_pareto.png"

ROBOTICS_FIGURE = ROOT / "figures" / "robotics_compression_pareto.png"


def load_json(
    path: Path,
) -> dict:
    if not path.exists():
        st.error(f"Missing evidence file: {path}")
        st.stop()

    return json.loads(path.read_text(encoding="utf-8"))


st.set_page_config(
    page_title=("Q-VLA Forge — Compression Evidence"),
    layout="wide",
)

st.title("Sprint 2 — Compression Evidence")

st.caption(
    "Classical INT8/SVD versus quantum-inspired "
    "TT/MPS compression on the shared VLA pilot."
)

evidence = load_json(EVIDENCE_PATH)

methods = evidence["methods"]

table_rows = []

for method in methods:
    table_rows.append(
        {
            "Domain": method["domain"],
            "Method": method["method"],
            "Configuration": method["configuration"],
            "Compression": (
                f"{method['compression_ratio_mean']:.3f} "
                f"± {method['compression_ratio_std']:.3f}×"
            ),
            "MSE Δ": (
                f"{method['mse_change_mean']:.3f} " f"± {method['mse_change_std']:.3f}%"
            ),
            "MAE Δ": (
                f"{method['mae_change_mean']:.3f} " f"± {method['mae_change_std']:.3f}%"
            ),
            "Feasible Runs": (f"{method['pilot_feasible_runs']}/3"),
            "Pareto": ("Yes" if method["pareto_efficient"] else "No"),
        }
    )

st.subheader("Three-Seed Validation")

st.dataframe(
    pd.DataFrame(table_rows),
    use_container_width=True,
    hide_index=True,
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Validation Seeds",
        len(evidence["seeds"]),
    )

with col2:
    st.metric(
        "Target Ablation Points",
        evidence["ablation_points"],
    )

st.subheader("Compression–Error Pareto Analysis")

left, right = st.columns(2)

with left:
    st.markdown("### Autonomous Driving")

    if DRIVING_FIGURE.exists():
        st.image(
            str(DRIVING_FIGURE),
            use_container_width=True,
        )

    st.write(
        "Pareto frontier:",
        evidence["driving_pareto_frontier"],
    )

with right:
    st.markdown("### Robotics")

    if ROBOTICS_FIGURE.exists():
        st.image(
            str(ROBOTICS_FIGURE),
            use_container_width=True,
        )

    st.write(
        "Pareto frontier:",
        evidence["robotics_pareto_frontier"],
    )

st.subheader("Classical vs Quantum-Inspired Ablation")

ablation = load_json(ABLATION_PATH)

ablation_rows = []

for point in ablation["points"]:
    ablation_rows.append(
        {
            "Domain": point["domain"],
            "Method": point["method"],
            "Target": point["target"],
            "Selected Layers": len(point["selected_layers"]),
            "Compressed Layers": len(point["compressed_layers"]),
            "Compression Ratio": round(
                point["compression_ratio"],
                4,
            ),
            "MSE Δ (%)": round(
                point["mse_change_percent"],
                4,
            ),
            "MAE Δ (%)": round(
                point["mae_change_percent"],
                4,
            ),
        }
    )

st.dataframe(
    pd.DataFrame(ablation_rows),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Quantum-Inspired Method")

st.write(evidence["quantum_inspired_method"])

st.write(
    "Quantum hardware used:",
    evidence["quantum_hardware_used"],
)

st.subheader("Proposal-Supported Claims")

for claim in evidence["supported_claims"]:
    st.markdown(f"- {claim}")

st.subheader("Limitations")

for limitation in evidence["limitations"]:
    st.markdown(f"- {limitation}")
