"""Matched-classical PPO training using the frozen Sprint 4 PPO protocol."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch

from q_vla_forge.rl.matched_classical_policy import (
    MatchedClassicalActorCritic,
)
from q_vla_forge.rl.ppo import (
    DEFAULT_PPO_CONFIG,
    PPOConfig,
    PPOEvaluation,
    RolloutBuffer,
    _ppo_update,
    evaluate_policy,
    set_ppo_seed,
)


@dataclass(frozen=True)
class MatchedClassicalPPOTrainingResult:
    """Evidence returned by one matched-classical PPO run."""

    seed: int
    domain: str

    evaluations: tuple[PPOEvaluation, ...]

    total_environment_steps: int
    completed_training_episodes: int

    projection_parameters: int
    core_parameters: int
    action_head_parameters: int
    actor_parameters: int
    critic_parameters: int
    total_parameters: int

    training_seconds: float


def train_matched_classical_ppo(
    *,
    domain: str,
    seed: int,
    environment_factory: Callable[
        [],
        gym.Env[Any, Any],
    ],
    evaluation_seeds: tuple[int, ...],
    config: PPOConfig = DEFAULT_PPO_CONFIG,
    progress_callback: (
        Callable[
            [PPOEvaluation],
            None,
        ]
        | None
    ) = None,
) -> MatchedClassicalPPOTrainingResult:
    """Train the parameter-matched classical actor under frozen PPO."""

    set_ppo_seed(seed)

    env = environment_factory()

    if not isinstance(
        env.observation_space,
        gym.spaces.Box,
    ):
        raise TypeError("matched-classical PPO requires a Box observation space")

    if not isinstance(
        env.action_space,
        gym.spaces.Box,
    ):
        raise TypeError("matched-classical PPO requires a Box action space")

    if env.observation_space.shape is None:
        raise TypeError("matched-classical PPO requires a shaped observation space")

    observation, _ = env.reset(seed=seed)

    env.action_space.seed(seed)

    observation_dim = int(np.prod(env.observation_space.shape))

    action_low = np.asarray(
        env.action_space.low,
        dtype=np.float32,
    )

    action_high = np.asarray(
        env.action_space.high,
        dtype=np.float32,
    )

    model = MatchedClassicalActorCritic(
        observation_dim=observation_dim,
        action_low=action_low,
        action_high=action_high,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    generator = torch.Generator().manual_seed(seed)

    evaluations: list[PPOEvaluation] = []

    completed_training_episodes = 0
    environment_steps = 0

    def record_evaluation() -> None:
        mean_reward, reward_sd, success_rate = evaluate_policy(
            model=cast(
                Any,
                model,
            ),
            environment_factory=environment_factory,
            seeds=evaluation_seeds,
        )

        evaluation = PPOEvaluation(
            environment_steps=environment_steps,
            completed_training_episodes=(completed_training_episodes),
            mean_reward=mean_reward,
            reward_standard_deviation=reward_sd,
            success_rate=success_rate,
        )

        evaluations.append(evaluation)

        if progress_callback is not None:
            progress_callback(evaluation)

    record_evaluation()

    next_evaluation = config.evaluation_frequency_steps

    started = time.perf_counter()

    while environment_steps < config.total_environment_steps:
        steps_until_evaluation = next_evaluation - environment_steps

        steps_remaining = config.total_environment_steps - environment_steps

        rollout_length = min(
            config.rollout_steps,
            steps_until_evaluation,
            steps_remaining,
        )

        buffer = RolloutBuffer.empty()

        for _ in range(rollout_length):
            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
            ).unsqueeze(0)

            with torch.no_grad():
                (
                    raw_action_tensor,
                    log_probability_tensor,
                    value_tensor,
                ) = model.sample_action(observation_tensor)

                environment_action_tensor = model.scale_raw_action(raw_action_tensor)

            raw_action = raw_action_tensor[0].cpu().numpy()

            environment_action = environment_action_tensor[0].cpu().numpy()

            (
                next_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(environment_action)

            done = bool(terminated or truncated)

            buffer.observations.append(
                np.asarray(
                    observation,
                    dtype=np.float32,
                )
            )

            buffer.raw_actions.append(raw_action)

            buffer.log_probabilities.append(float(log_probability_tensor[0].item()))

            buffer.rewards.append(float(reward))

            buffer.dones.append(done)

            buffer.values.append(float(value_tensor[0].item()))

            environment_steps += 1

            observation = next_observation

            if done:
                completed_training_episodes += 1

                observation, _ = env.reset()

        if buffer.dones[-1]:
            bootstrap_value = 0.0

        else:
            with torch.no_grad():
                bootstrap_value = float(
                    model.value(
                        torch.as_tensor(
                            observation,
                            dtype=torch.float32,
                        ).unsqueeze(0)
                    )[0].item()
                )

        _ppo_update(
            model=cast(
                Any,
                model,
            ),
            optimizer=optimizer,
            buffer=buffer,
            bootstrap_value=bootstrap_value,
            config=config,
            generator=generator,
        )

        if environment_steps == next_evaluation:
            record_evaluation()

            next_evaluation += config.evaluation_frequency_steps

    training_seconds = time.perf_counter() - started

    env.close()

    return MatchedClassicalPPOTrainingResult(
        seed=seed,
        domain=domain,
        evaluations=tuple(evaluations),
        total_environment_steps=(environment_steps),
        completed_training_episodes=(completed_training_episodes),
        projection_parameters=(model.projection_parameter_count),
        core_parameters=(model.core_parameter_count),
        action_head_parameters=(model.action_head_parameter_count),
        actor_parameters=(model.actor_parameter_count),
        critic_parameters=(model.critic_parameter_count),
        total_parameters=(model.total_parameter_count),
        training_seconds=(training_seconds),
    )
