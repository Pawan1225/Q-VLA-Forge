"""Sprint 7.3 — frozen final baseline summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASELINE_MANIFEST = "results/sprint1-baseline-manifest.json"

BASELINE_SUMMARY = "results/baseline-validation-summary.json"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")

    return payload


def metric_summary(
    domain_payload: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    value = domain_payload.get(metric)

    if not isinstance(value, dict):
        raise KeyError(f"Missing baseline metric: {metric}")

    return {
        "mean": value["mean"],
        "std": value["std"],
        "values": value.get("values", []),
    }


def build_final_baseline(
    root: Path,
) -> dict[str, Any]:
    """Build the canonical Sprint 7.3 baseline summary."""

    manifest_path = root / BASELINE_MANIFEST
    summary_path = root / BASELINE_SUMMARY

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    manifest = load_json(manifest_path)
    summary = load_json(summary_path)

    expected_seeds = (42, 123, 456)

    manifest_seeds = tuple(manifest.get("seeds", []))

    summary_seeds = tuple(summary.get("seeds", []))

    if manifest_seeds != expected_seeds:
        raise ValueError("Baseline manifest does not use " "locked seeds 42, 123, 456.")

    if summary_seeds != expected_seeds:
        raise ValueError("Baseline summary does not use " "locked seeds 42, 123, 456.")

    domains: dict[str, Any] = {}

    for key in ("driving", "robotics"):
        domain_payload = summary.get(key)

        if not isinstance(
            domain_payload,
            dict,
        ):
            raise KeyError(f"Missing baseline domain: {key}")

        domain_seeds = tuple(domain_payload.get("seeds", []))

        if domain_seeds != expected_seeds:
            raise ValueError(f"{key} baseline seed mismatch.")

        domains[key] = {
            "domain": domain_payload["domain"],
            "seeds": list(domain_seeds),
            "test_mse": metric_summary(
                domain_payload,
                "test_mse",
            ),
            "test_mae": metric_summary(
                domain_payload,
                "test_mae",
            ),
            "mean_latency_ms": metric_summary(
                domain_payload,
                "mean_latency_ms",
            ),
            "p95_latency_ms": metric_summary(
                domain_payload,
                "p95_latency_ms",
            ),
            "parameters": domain_payload["parameters"],
            "fp32_model_size_bytes": (domain_payload["fp32_model_size_bytes"]),
        }

    return {
        "sprint": "7.3",
        "protocol": "final_baseline_summary",
        "status": "FROZEN",
        "baseline_name": manifest["baseline_name"],
        "baseline_version": manifest["baseline_version"],
        "required_seeds": list(expected_seeds),
        "architecture": {
            "vision_dim": manifest["vision_dim"],
            "language_dim": manifest["language_dim"],
            "state_dim": manifest["state_dim"],
            "fusion_dim": manifest["fusion_dim"],
            "latent_dim": manifest["latent_dim"],
            "action_dim": manifest["action_dim"],
            "parameters": manifest["parameters"],
            "fp32_model_size_bytes": (manifest["fp32_model_size_bytes"]),
        },
        "training_protocol": {
            "optimizer": manifest["optimizer"],
            "scheduler": manifest["scheduler"],
            "loss": manifest["loss"],
            "epochs": manifest["epochs"],
            "batch_size": manifest["batch_size"],
            "train_size": manifest["train_size"],
            "validation_size": manifest["validation_size"],
            "test_size": manifest["test_size"],
        },
        "domains": domains,
        "hardware_note": (
            "Sprint 7.3 preserves the latency "
            "measurements recorded by the frozen "
            "Sprint 1 artifacts. No hardware "
            "configuration is inferred where the "
            "baseline summary does not explicitly "
            "record it."
        ),
        "provenance": {
            "manifest": BASELINE_MANIFEST,
            "summary": BASELINE_SUMMARY,
            "evidence_files": manifest["evidence_files"],
        },
    }
