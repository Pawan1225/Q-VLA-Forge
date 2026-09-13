from __future__ import annotations

import pytest

from q_vla_forge.evaluation.rl_environment_audit import (
    AUDIT_SEEDS,
    EpisodeAuditRecord,
    heuristic_beats_random,
    summarize_episode_records,
)


def test_audit_seed_bank() -> None:
    assert len(AUDIT_SEEDS) == 100

    assert AUDIT_SEEDS[0] == 10_000

    assert AUDIT_SEEDS[-1] == 10_099


def test_episode_record_rejects_non_finite_reward() -> None:
    with pytest.raises(ValueError):
        EpisodeAuditRecord(
            domain="robotics",
            policy="random",
            seed=10_000,
            total_reward=float("nan"),
            episode_length=10,
            success=False,
            terminated=False,
            truncated=True,
        )


def test_episode_record_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError):
        EpisodeAuditRecord(
            domain="robotics",
            policy="random",
            seed=10_000,
            total_reward=1.0,
            episode_length=0,
            success=False,
            terminated=False,
            truncated=True,
        )


def test_episode_record_rejects_double_end_state() -> None:
    with pytest.raises(ValueError):
        EpisodeAuditRecord(
            domain="robotics",
            policy="random",
            seed=10_000,
            total_reward=1.0,
            episode_length=10,
            success=False,
            terminated=True,
            truncated=True,
        )


def test_summary_statistics() -> None:
    records = [
        EpisodeAuditRecord(
            domain="robotics",
            policy="heuristic",
            seed=10_000,
            total_reward=10.0,
            episode_length=20,
            success=True,
            terminated=True,
            truncated=False,
        ),
        EpisodeAuditRecord(
            domain="robotics",
            policy="heuristic",
            seed=10_001,
            total_reward=12.0,
            episode_length=22,
            success=True,
            terminated=True,
            truncated=False,
        ),
    ]

    summary = summarize_episode_records(records)

    assert summary.domain == "robotics"
    assert summary.policy == "heuristic"
    assert summary.episodes == 2
    assert summary.mean_reward == 11.0
    assert summary.success_rate == 1.0
    assert summary.mean_episode_length == 21.0
    assert summary.terminated_rate == 1.0
    assert summary.truncated_rate == 0.0


def test_single_record_summary_has_zero_sd() -> None:
    summary = summarize_episode_records(
        [
            EpisodeAuditRecord(
                domain="robotics",
                policy="random",
                seed=10_000,
                total_reward=2.0,
                episode_length=100,
                success=False,
                terminated=False,
                truncated=True,
            )
        ]
    )

    assert summary.reward_standard_deviation == 0.0


def test_summary_rejects_empty_records() -> None:
    with pytest.raises(ValueError):
        summarize_episode_records([])


def test_summary_rejects_mixed_domains() -> None:
    records = [
        EpisodeAuditRecord(
            domain="robotics",
            policy="random",
            seed=10_000,
            total_reward=0.0,
            episode_length=10,
            success=False,
            terminated=False,
            truncated=True,
        ),
        EpisodeAuditRecord(
            domain="autonomous_driving",
            policy="random",
            seed=10_001,
            total_reward=0.0,
            episode_length=10,
            success=False,
            terminated=False,
            truncated=True,
        ),
    ]

    with pytest.raises(ValueError):
        summarize_episode_records(records)


def test_summary_rejects_mixed_policies() -> None:
    records = [
        EpisodeAuditRecord(
            domain="robotics",
            policy="random",
            seed=10_000,
            total_reward=0.0,
            episode_length=10,
            success=False,
            terminated=False,
            truncated=True,
        ),
        EpisodeAuditRecord(
            domain="robotics",
            policy="heuristic",
            seed=10_001,
            total_reward=10.0,
            episode_length=20,
            success=True,
            terminated=True,
            truncated=False,
        ),
    ]

    with pytest.raises(ValueError):
        summarize_episode_records(records)


def test_heuristic_comparison() -> None:
    random_summary = summarize_episode_records(
        [
            EpisodeAuditRecord(
                domain="robotics",
                policy="random",
                seed=10_000,
                total_reward=1.0,
                episode_length=100,
                success=False,
                terminated=False,
                truncated=True,
            )
        ]
    )

    heuristic_summary = summarize_episode_records(
        [
            EpisodeAuditRecord(
                domain="robotics",
                policy="heuristic",
                seed=10_000,
                total_reward=10.0,
                episode_length=20,
                success=True,
                terminated=True,
                truncated=False,
            )
        ]
    )

    assert heuristic_beats_random(
        heuristic=heuristic_summary,
        random=random_summary,
    )


def test_heuristic_comparison_rejects_domain_mismatch() -> None:
    robotics_summary = summarize_episode_records(
        [
            EpisodeAuditRecord(
                domain="robotics",
                policy="heuristic",
                seed=10_000,
                total_reward=10.0,
                episode_length=20,
                success=True,
                terminated=True,
                truncated=False,
            )
        ]
    )

    driving_summary = summarize_episode_records(
        [
            EpisodeAuditRecord(
                domain="autonomous_driving",
                policy="random",
                seed=10_000,
                total_reward=1.0,
                episode_length=100,
                success=False,
                terminated=False,
                truncated=True,
            )
        ]
    )

    with pytest.raises(ValueError):
        heuristic_beats_random(
            heuristic=robotics_summary,
            random=driving_summary,
        )
