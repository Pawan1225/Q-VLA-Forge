"""Compact PyTorch PPO implementation for Sprint 4."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class PPOConfig:
    """Frozen classical PPO configuration."""

    total_environment_steps: int = 20_000
    evaluation_frequency_steps: int = 1_000
    evaluation_episodes: int = 20

    rollout_steps: int = 256

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_coefficient: float = 0.20

    learning_rate: float = 3e-4

    update_epochs: int = 10
    minibatch_size: int = 64

    value_coefficient: float = 0.50
    entropy_coefficient: float = 0.01

    max_gradient_norm: float = 0.50

    actor_hidden_dim: int = 32
    critic_hidden_dim: int = 32

    initial_log_std: float = -0.5

    def __post_init__(self) -> None:
        if self.total_environment_steps != 20_000:
            raise ValueError("Sprint 4 PPO budget is frozen to 20,000 steps")

        if self.evaluation_frequency_steps != 1_000:
            raise ValueError("evaluation frequency is frozen to 1,000 steps")

        if self.evaluation_episodes != 20:
            raise ValueError("evaluation episode count is frozen to 20")

        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")

        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")

        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")


DEFAULT_PPO_CONFIG = PPOConfig()


@dataclass(frozen=True)
class PPOEvaluation:
    """One deterministic held-out PPO evaluation."""

    environment_steps: int
    completed_training_episodes: int
    mean_reward: float
    reward_standard_deviation: float
    success_rate: float


@dataclass(frozen=True)
class PPOTrainingResult:
    """Evidence returned by one principal PPO run."""

    seed: int
    domain: str
    evaluations: tuple[PPOEvaluation, ...]
    total_environment_steps: int
    completed_training_episodes: int
    actor_parameters: int
    critic_parameters: int
    total_parameters: int
    training_seconds: float


@dataclass
class RolloutBuffer:
    """Temporary PPO rollout storage."""

    observations: list[np.ndarray]
    raw_actions: list[np.ndarray]
    log_probabilities: list[float]
    rewards: list[float]
    dones: list[bool]
    values: list[float]

    @classmethod
    def empty(cls) -> RolloutBuffer:
        return cls(
            observations=[],
            raw_actions=[],
            log_probabilities=[],
            rewards=[],
            dones=[],
            values=[],
        )


class GaussianActorCritic(nn.Module):
    """Small continuous-action actor-critic network."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        actor_hidden_dim: int = 32,
        critic_hidden_dim: int = 32,
        initial_log_std: float = -0.5,
    ) -> None:
        super().__init__()

        action_low = np.asarray(
            action_low,
            dtype=np.float32,
        )

        action_high = np.asarray(
            action_high,
            dtype=np.float32,
        )

        if action_low.shape != action_high.shape:
            raise ValueError("action bounds must have matching shapes")

        if np.any(action_high <= action_low):
            raise ValueError("every action upper bound must exceed lower bound")

        self.action_dim = int(action_low.shape[0])

        self.actor_body = nn.Sequential(
            nn.Linear(
                observation_dim,
                actor_hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                actor_hidden_dim,
                actor_hidden_dim,
            ),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(
            actor_hidden_dim,
            self.action_dim,
        )

        self.log_std = nn.Parameter(
            torch.full(
                (self.action_dim,),
                float(initial_log_std),
            )
        )

        self.critic_body = nn.Sequential(
            nn.Linear(
                observation_dim,
                critic_hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                critic_hidden_dim,
                critic_hidden_dim,
            ),
            nn.Tanh(),
        )

        self.value_head = nn.Linear(
            critic_hidden_dim,
            1,
        )

        self.register_buffer(
            "action_low",
            torch.as_tensor(
                action_low,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "action_high",
            torch.as_tensor(
                action_high,
                dtype=torch.float32,
            ),
        )

    def distribution(
        self,
        observations: torch.Tensor,
    ) -> Normal:
        features = self.actor_body(observations)

        mean = self.actor_mean(features)

        std = torch.exp(self.log_std).expand_as(mean)

        return Normal(
            mean,
            std,
        )

    def value(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        features = self.critic_body(observations)

        return self.value_head(features).squeeze(-1)

    def scale_raw_action(
        self,
        raw_action: torch.Tensor,
    ) -> torch.Tensor:
        squashed = torch.tanh(raw_action)

        action_low = cast(
            torch.Tensor,
            self.action_low,
        )

        action_high = cast(
            torch.Tensor,
            self.action_high,
        )

        return action_low + 0.5 * (squashed + 1.0) * (action_high - action_low)

    def sample_action(
        self,
        observation: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        distribution = self.distribution(observation)

        raw_action = distribution.sample()

        log_probability = distribution.log_prob(raw_action).sum(dim=-1)

        value = self.value(observation)

        return (
            raw_action,
            log_probability,
            value,
        )

    def deterministic_action(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        distribution = self.distribution(observation)

        return self.scale_raw_action(distribution.mean)


def set_ppo_seed(
    seed: int,
) -> None:
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


def count_actor_parameters(
    model: GaussianActorCritic,
) -> int:
    actor_parameters = list(model.actor_body.parameters())

    actor_parameters.extend(model.actor_mean.parameters())

    actor_parameters.append(model.log_std)

    return sum(parameter.numel() for parameter in actor_parameters)


def count_critic_parameters(
    model: GaussianActorCritic,
) -> int:
    parameters = list(model.critic_body.parameters())

    parameters.extend(model.value_head.parameters())

    return sum(parameter.numel() for parameter in parameters)


def _compute_gae(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    advantages = torch.zeros_like(rewards)

    last_advantage = torch.tensor(
        0.0,
        dtype=torch.float32,
    )

    for index in reversed(range(rewards.shape[0])):
        if index == rewards.shape[0] - 1:
            next_value = bootstrap_value
        else:
            next_value = values[index + 1]

        next_non_terminal = 1.0 - dones[index]

        delta = rewards[index] + gamma * next_value * next_non_terminal - values[index]

        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage

        advantages[index] = last_advantage

    returns = advantages + values

    return (
        advantages,
        returns,
    )


def _ppo_update(
    *,
    model: GaussianActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    bootstrap_value: float,
    config: PPOConfig,
    generator: torch.Generator,
) -> None:
    observations = torch.as_tensor(
        np.asarray(buffer.observations),
        dtype=torch.float32,
    )

    raw_actions = torch.as_tensor(
        np.asarray(buffer.raw_actions),
        dtype=torch.float32,
    )

    old_log_probabilities = torch.as_tensor(
        buffer.log_probabilities,
        dtype=torch.float32,
    )

    rewards = torch.as_tensor(
        buffer.rewards,
        dtype=torch.float32,
    )

    dones = torch.as_tensor(
        buffer.dones,
        dtype=torch.float32,
    )

    values = torch.as_tensor(
        buffer.values,
        dtype=torch.float32,
    )

    advantages, returns = _compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        bootstrap_value=torch.tensor(
            bootstrap_value,
            dtype=torch.float32,
        ),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    advantage_std = advantages.std(unbiased=False)

    advantages = (advantages - advantages.mean()) / (advantage_std + 1e-8)

    sample_count = observations.shape[0]

    for _ in range(config.update_epochs):
        permutation = torch.randperm(
            sample_count,
            generator=generator,
        )

        for start in range(
            0,
            sample_count,
            config.minibatch_size,
        ):
            indexes = permutation[start : start + config.minibatch_size]

            batch_observations = observations[indexes]

            batch_raw_actions = raw_actions[indexes]

            batch_old_log_probabilities = old_log_probabilities[indexes]

            batch_advantages = advantages[indexes]

            batch_returns = returns[indexes]

            distribution = model.distribution(batch_observations)

            new_log_probabilities = distribution.log_prob(batch_raw_actions).sum(dim=-1)

            entropy = distribution.entropy().sum(dim=-1).mean()

            ratios = torch.exp(new_log_probabilities - batch_old_log_probabilities)

            unclipped = ratios * batch_advantages

            clipped = (
                torch.clamp(
                    ratios,
                    1.0 - config.clip_coefficient,
                    1.0 + config.clip_coefficient,
                )
                * batch_advantages
            )

            policy_loss = -torch.min(
                unclipped,
                clipped,
            ).mean()

            predicted_values = model.value(batch_observations)

            value_loss = torch.mean((predicted_values - batch_returns) ** 2)

            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_gradient_norm,
            )

            optimizer.step()


def evaluate_policy(
    *,
    model: GaussianActorCritic,
    environment_factory: Callable[
        [],
        gym.Env[Any, Any],
    ],
    seeds: tuple[int, ...],
) -> tuple[
    float,
    float,
    float,
]:
    rewards: list[float] = []
    successes: list[float] = []

    model.eval()

    with torch.no_grad():
        for seed in seeds:
            env = environment_factory()

            observation, _ = env.reset(seed=seed)

            total_reward = 0.0

            while True:
                observation_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                ).unsqueeze(0)

                action = model.deterministic_action(observation_tensor)[0].cpu().numpy()

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                total_reward += float(reward)

                if terminated or truncated:
                    rewards.append(total_reward)

                    successes.append(float(info["success"]))

                    env.close()
                    break

    model.train()

    reward_array = np.asarray(
        rewards,
        dtype=np.float64,
    )

    return (
        float(np.mean(reward_array)),
        float(
            np.std(
                reward_array,
                ddof=1,
            )
            if len(reward_array) > 1
            else 0.0
        ),
        float(np.mean(successes)),
    )


def train_ppo(
    *,
    domain: str,
    seed: int,
    environment_factory: Callable[
        [],
        gym.Env[Any, Any],
    ],
    evaluation_seeds: tuple[int, ...],
    config: PPOConfig = DEFAULT_PPO_CONFIG,
    checkpoint_path: str | None = None,
) -> PPOTrainingResult:
    """Train one frozen-budget classical PPO reference."""
    set_ppo_seed(seed)

    env = environment_factory()

    observation, _ = env.reset(seed=seed)

    env.action_space.seed(seed)

    observation_shape = env.observation_space.shape

    if observation_shape is None:
        raise ValueError("PPO requires a shaped observation space")

    observation_dim = int(math.prod(observation_shape))

    action_space = env.action_space

    if not isinstance(
        action_space,
        gym.spaces.Box,
    ):
        raise TypeError("PPO requires a continuous Box action space")

    action_low = np.asarray(
        action_space.low,
        dtype=np.float32,
    )

    action_high = np.asarray(
        action_space.high,
        dtype=np.float32,
    )

    model = GaussianActorCritic(
        observation_dim=observation_dim,
        action_low=action_low,
        action_high=action_high,
        actor_hidden_dim=config.actor_hidden_dim,
        critic_hidden_dim=config.critic_hidden_dim,
        initial_log_std=config.initial_log_std,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    generator = torch.Generator().manual_seed(seed)

    actor_parameters = count_actor_parameters(model)

    critic_parameters = count_critic_parameters(model)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    evaluations: list[PPOEvaluation] = []

    completed_training_episodes = 0
    environment_steps = 0

    def record_evaluation() -> None:
        mean_reward, reward_sd, success_rate = evaluate_policy(
            model=model,
            environment_factory=environment_factory,
            seeds=evaluation_seeds,
        )

        evaluations.append(
            PPOEvaluation(
                environment_steps=environment_steps,
                completed_training_episodes=(completed_training_episodes),
                mean_reward=mean_reward,
                reward_standard_deviation=reward_sd,
                success_rate=success_rate,
            )
        )

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
            model=model,
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

    if checkpoint_path is not None:
        torch.save(
            {
                "schema_version": 1,
                "policy_family": "classical_ppo",
                "domain": domain,
                "seed": seed,
                "checkpoint_selection": "final_20000_step_policy",
                "total_environment_steps": environment_steps,
                "observation_dim": observation_dim,
                "action_low": action_low.tolist(),
                "action_high": action_high.tolist(),
                "actor_hidden_dim": config.actor_hidden_dim,
                "critic_hidden_dim": config.critic_hidden_dim,
                "initial_log_std": config.initial_log_std,
                "model_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )

    return PPOTrainingResult(
        seed=seed,
        domain=domain,
        evaluations=tuple(evaluations),
        total_environment_steps=environment_steps,
        completed_training_episodes=(completed_training_episodes),
        actor_parameters=actor_parameters,
        critic_parameters=critic_parameters,
        total_parameters=total_parameters,
        training_seconds=training_seconds,
    )
