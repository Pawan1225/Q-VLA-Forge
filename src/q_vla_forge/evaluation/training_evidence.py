"""Proposal-evidence utilities for Sprint 3 training efficiency."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    """One evidence-backed proposal claim."""

    claim_id: str
    category: str
    supported: bool
    statement: str
    evidence_basis: str


def method_label(
    method: str,
) -> str:
    """Return human-readable method label."""
    labels = {
        "trainable_svd": "Trainable SVD",
        "trainable_tt_mps": "Trainable TT/MPS",
    }

    try:
        return labels[method]
    except KeyError as exc:
        raise ValueError(f"unsupported method: {method}") from exc


def domain_label(
    domain: str,
) -> str:
    """Return human-readable domain label."""
    labels = {
        "autonomous_driving": "autonomous driving",
        "robotics": "robotics",
    }

    try:
        return labels[domain]
    except KeyError as exc:
        raise ValueError(f"unsupported domain: {domain}") from exc


def build_method_claim(
    *,
    domain: str,
    method: str,
    reached: int,
    total: int,
    step_reduction_mean: float | None,
    step_reduction_std: float | None,
    robust_ten_percent: bool,
) -> Claim:
    """Build one domain/method training-efficiency claim."""
    readable_domain = domain_label(domain)

    readable_method = method_label(method)

    claim_id = f"{domain}-" f"{method}-" "training-efficiency"

    if robust_ten_percent:
        if (
            reached != total
            or step_reduction_mean is None
            or step_reduction_std is None
        ):
            raise ValueError("robust efficiency claim has " "inconsistent evidence")

        return Claim(
            claim_id=claim_id,
            category="training_efficiency",
            supported=True,
            statement=(
                f"{readable_method} reached the paired "
                f"FP32 target in {reached}/{total} seeds "
                f"for {readable_domain} while reducing "
                f"optimizer steps by "
                f"{step_reduction_mean:.2f} ± "
                f"{step_reduction_std:.2f}% on average."
            ),
            evidence_basis=(
                "Three deterministic seeds; all targets "
                "reached; mean optimizer-step reduction "
                "met or exceeded the predefined 10% threshold."
            ),
        )

    if reached == total:
        if step_reduction_mean is None:
            statement = (
                f"{readable_method} reached the paired "
                f"FP32 target in all {total} seeds for "
                f"{readable_domain}, but no valid "
                "optimizer-step reduction estimate "
                "was available."
            )
        else:
            std = 0.0 if step_reduction_std is None else step_reduction_std

            statement = (
                f"{readable_method} reached the paired "
                f"FP32 target in all {total} seeds for "
                f"{readable_domain}; the mean optimizer-step "
                f"reduction was {step_reduction_mean:.2f} ± "
                f"{std:.2f}%, below the predefined robust "
                "10% criterion."
            )

        return Claim(
            claim_id=claim_id,
            category="training_efficiency",
            supported=True,
            statement=statement,
            evidence_basis=(
                "Three-seed validation completed, but the "
                "predefined robust >=10% efficiency rule "
                "was not satisfied."
            ),
        )

    return Claim(
        claim_id=claim_id,
        category="training_efficiency",
        supported=True,
        statement=(
            f"{readable_method} reached the paired FP32 "
            f"target in {reached}/{total} seeds for "
            f"{readable_domain}; therefore the experiment "
            "does not support a robust three-seed "
            "training-efficiency claim."
        ),
        evidence_basis=(
            "Target reach was incomplete across the " "three deterministic seeds."
        ),
    )


def build_cross_domain_claim(
    *,
    method: str,
    robust_cross_domain: bool,
    target_consistency: str,
    efficiency_consistency: str,
) -> Claim:
    """Build one cross-domain transfer claim."""
    readable_method = method_label(method)

    if robust_cross_domain:
        return Claim(
            claim_id=(f"{method}-cross-domain-efficiency"),
            category="cross_domain",
            supported=True,
            statement=(
                f"{readable_method} satisfied the predefined "
                "robust training-efficiency criterion in both "
                "autonomous-driving and robotics proxy tasks."
            ),
            evidence_basis=(
                "Both domains passed the three-seed "
                "all-target-reached and >=10% mean "
                "optimizer-step reduction rule."
            ),
        )

    return Claim(
        claim_id=(f"{method}-cross-domain-efficiency"),
        category="cross_domain",
        supported=True,
        statement=(
            f"{readable_method} was architecturally reusable "
            "across both domains, but did not demonstrate "
            "robust training-efficiency improvement in both; "
            f"target consistency was {target_consistency} "
            f"and efficiency consistency was "
            f"{efficiency_consistency}."
        ),
        evidence_basis=(
            "Cross-domain comparison of the frozen " "three-seed experiments."
        ),
    )


def prohibited_claims() -> tuple[Claim, ...]:
    """Return claims explicitly unsupported by Sprint 3."""
    return (
        Claim(
            claim_id="quantum-advantage",
            category="claim_control",
            supported=False,
            statement=("Sprint 3 does not establish quantum advantage."),
            evidence_basis=(
                "TT/MPS is a quantum-inspired tensor-network "
                "method executed classically."
            ),
        ),
        Claim(
            claim_id="quantum-speedup",
            category="claim_control",
            supported=False,
            statement=("Sprint 3 does not establish quantum speedup."),
            evidence_basis=(
                "No quantum hardware or quantum runtime " "comparison was performed."
            ),
        ),
        Claim(
            claim_id="native-tt-runtime-speedup",
            category="claim_control",
            supported=False,
            statement=(
                "Sprint 3 does not establish native TT/MPS " "runtime acceleration."
            ),
            evidence_basis=(
                "The current TT/MPS forward path reconstructs "
                "a temporary differentiable dense-equivalent "
                "weight during classical execution."
            ),
        ),
        Claim(
            claim_id="production-vla-validation",
            category="claim_control",
            supported=False,
            statement=(
                "Sprint 3 does not establish performance on "
                "full-scale production 7B-class VLA models."
            ),
            evidence_basis=(
                "The pilot uses compact synthetic driving " "and robotics proxy tasks."
            ),
        ),
    )
