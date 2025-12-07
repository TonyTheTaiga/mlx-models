# https://medium.com/dabbler-in-de-stress/the-inverted-pendulum-problem-with-deep-reinforcement-learning-9f149b68c018


import argparse
import json
import math
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from env import Env

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
LOG_2PI = float(math.log(2.0 * math.pi))


class Actor(nn.Module):
    """Policy network"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, output_dim)
        self.log_std_layer = nn.Linear(128, output_dim)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        single_input = x.ndim == 1
        if single_input:
            x = mx.expand_dims(x, axis=0)
        x = mx.tanh(self.fc1(x))
        x = mx.tanh(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = mx.clip(self.log_std_layer(x), LOG_STD_MIN, LOG_STD_MAX)
        if single_input:
            mean = mx.squeeze(mean, axis=0)
            log_std = mx.squeeze(log_std, axis=0)
        return mean, log_std


class Critic(nn.Module):
    """Value network"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, 1)

    def __call__(self, x: mx.array) -> mx.array:
        single_input = x.ndim == 1
        if single_input:
            x = mx.expand_dims(x, axis=0)
        x = mx.tanh(self.fc1(x))
        x = mx.tanh(self.fc2(x))
        value = self.value_layer(x)
        if single_input:
            value = mx.squeeze(value, axis=0)
        return value


def sample_actions(mean: mx.array, log_std: mx.array) -> mx.array:
    """Sample Gaussian actions given mean and log std."""
    std = mx.exp(log_std)
    return mx.random.normal(mean.shape, loc=mean, scale=std, dtype=mean.dtype)


def gaussian_log_prob(actions: mx.array, mean: mx.array, log_std: mx.array) -> mx.array:
    """Compute log-probabilities of Gaussian actions."""
    var = mx.exp(2.0 * log_std)
    log_probs = -0.5 * (((actions - mean) ** 2) / var + 2.0 * log_std + LOG_2PI)
    return mx.sum(log_probs, axis=-1)


def gaussian_entropy(log_std: mx.array) -> mx.array:
    """Differential entropy of a diagonal Gaussian."""
    return mx.sum(log_std + 0.5 * (1.0 + LOG_2PI), axis=-1)


def collect_trajectories(
    actor: Actor, critic: Critic, env: Env, num_steps: int, last_obs: mx.array
):
    observations = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []
    thetas = []
    theta_dots = []

    obs = last_obs

    for _ in range(num_steps):
        observations.append(obs)
        mean, log_std = actor(obs)
        action = sample_actions(mean, log_std)
        log_prob = gaussian_log_prob(action, mean, log_std)
        value = critic(obs)
        obs, reward, done, info = env.step(action)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)
        values.append(value)
        thetas.append(info["theta"])
        theta_dots.append(info["theta_dot"])

        if done:
            obs = env.reset()

    bootstrap_value = critic(obs)

    return (
        mx.stack(observations),
        mx.stack(actions),
        mx.stack(log_probs),
        mx.array(rewards),
        mx.array(dones, dtype=mx.float32),
        mx.stack(values).squeeze(),
        bootstrap_value.squeeze(),
        obs,
        mx.array(thetas),
        mx.array(theta_dots),
    )


def compute_gae(
    rewards: mx.array,
    values: mx.array,
    bootstrap_value: mx.array,
    dones: mx.array,
    gamma: float,
    gae_lambda: float,
):
    advantages = mx.zeros_like(rewards)
    last_advantage = 0.0
    values = mx.concatenate([values, mx.reshape(bootstrap_value, (1,))], axis=0)

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]

    returns = advantages + values[:-1]
    return advantages, returns


def train(
    env: Env,
    actor: Actor,
    critic: Critic,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    num_iterations: int,
    num_steps: int,
    num_epochs: int,
    batch_size: int,
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    entropy_coef: float,
    checkpoint_dir: Path | None = None,
):
    obs = env.reset()

    for i in range(num_iterations):
        (
            observations,
            actions,
            log_probs,
            rewards,
            dones,
            values,
            bootstrap_value,
            obs,
            thetas,
            theta_dots,
        ) = collect_trajectories(actor, critic, env, num_steps, obs)

        advantages, returns = compute_gae(
            rewards, values, bootstrap_value, dones, gamma, gae_lambda
        )
        advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)

        # Update policy and value function
        for _ in range(num_epochs):
            indices = list(range(num_steps))
            random.shuffle(indices)

            for start in range(0, num_steps, batch_size):
                end = start + batch_size
                if end > num_steps:
                    end = num_steps

                batch_indices = indices[start:end]
                mx_indices = mx.array(batch_indices, dtype=mx.int32)
                batch_obs = mx.take(observations, mx_indices, axis=0)
                batch_actions = mx.take(actions, mx_indices, axis=0)
                batch_log_probs = mx.take(log_probs, mx_indices, axis=0)
                batch_advantages = mx.take(advantages, mx_indices, axis=0)
                batch_returns = mx.take(returns, mx_indices, axis=0)

                def actor_closure(model, obs, act, lp, adv):
                    mean, log_std = model(obs)
                    new_log_probs = gaussian_log_prob(act, mean, log_std)
                    ratio = mx.exp(new_log_probs - lp)
                    surr1 = ratio * adv
                    surr2 = mx.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                    actor_loss = -mx.mean(mx.minimum(surr1, surr2))
                    entropy = mx.mean(gaussian_entropy(log_std))
                    return actor_loss - entropy_coef * entropy

                actor_grad_fn = nn.value_and_grad(actor, actor_closure)
                actor_loss, actor_grads = actor_grad_fn(
                    actor,
                    batch_obs,
                    batch_actions,
                    batch_log_probs,
                    batch_advantages,
                )
                actor_optimizer.update(actor, actor_grads)

                def critic_closure(model, obs, ret):
                    return nn.losses.mse_loss(model(obs).squeeze(), ret)

                critic_grad_fn = nn.value_and_grad(critic, critic_closure)
                critic_loss, critic_grads = critic_grad_fn(critic, batch_obs, batch_returns)
                critic_optimizer.update(critic, critic_grads)

        mx.eval(
            actor.parameters(),
            critic.parameters(),
            actor_optimizer.state,
            critic_optimizer.state,
        )

        total_reward = rewards.sum().item()
        theta_abs_mean = mx.mean(mx.abs(thetas)).item()
        theta_dot_abs_mean = mx.mean(mx.abs(theta_dots)).item()
        print(
            "Iteration "
            f"{i}, Total Reward: {total_reward:.2f}, Actor Loss: {actor_loss.item():.4f}, "  # pyright: ignore
            f"Critic Loss: {critic_loss.item():.4f}, Mean|theta|: {theta_abs_mean:.4f}, "  # pyright: ignore
            f"Mean|theta_dot|: {theta_dot_abs_mean:.4f}"
        )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        actor_path = checkpoint_dir / "actor_latest.npz"
        critic_path = checkpoint_dir / "critic_latest.npz"
        actor.save_weights(str(actor_path))
        critic.save_weights(str(critic_path))
        print(f"Saved actor weights to {actor_path}")
        print(f"Saved critic weights to {critic_path}")

        env_config = {
            "m": float(env.m.item()),
            "l": float(env.l.item()),
            "g": float(env.g.item()),
            "dt": float(env.dt.item()),
            "max_torque": float(env.max_torque.item()),
            "theta_threshold": float(env.theta_threshold.item()),
            "max_theta_dot": float(env.max_theta_dot.item()),
            "init_theta_range": float(env.init_theta_range.item()),
            "init_theta_dot_range": float(env.init_theta_dot_range.item()),
        }
        training_config = {
            "num_iterations": num_iterations,
            "num_steps": num_steps,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_eps": clip_eps,
            "entropy_coef": entropy_coef,
        }
        config_data = {
            "artifacts": {
                "actor": actor_path.name,
                "critic": critic_path.name,
            },
            "env": env_config,
            "training": training_config,
        }
        config_path = checkpoint_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print(f"Saved training config to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training for Inverted Pendulum")
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=500,
        help="Number of training iterations.",
    )
    parser.add_argument("--num_steps", type=int, default=2048, help="Number of steps per rollout.")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs for PPO update."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for PPO update.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--max_torque",
        type=float,
        default=1.0,
        help="Maximum torque magnitude applied to the pendulum (smaller is harder).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("training/inv_pendulum/checkpoints"),
        help="Directory where trained weights will be stored.",
    )
    parser.add_argument(
        "--init_theta_range",
        type=float,
        default=math.pi,
        help="Initial theta is sampled uniformly from [-range, range].",
    )
    parser.add_argument(
        "--init_theta_dot_range",
        type=float,
        default=1.0,
        help="Initial angular velocity sampled uniformly from [-range, range].",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    mx.random.seed(args.seed)

    env = Env(
        m=2.0,
        l=1.0,
        max_torque=args.max_torque,
        init_theta_range=args.init_theta_range,
        init_theta_dot_range=args.init_theta_dot_range,
    )
    obs_dim = env.reset().shape[-1]
    action_dim = 1
    actor = Actor(obs_dim, action_dim)
    critic = Critic(obs_dim)
    mx.eval(actor.parameters(), critic.parameters())

    actor_optimizer = optim.Adam(learning_rate=args.lr)
    critic_optimizer = optim.Adam(learning_rate=args.lr)

    print("Starting PPO training...")
    train(
        env=env,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        num_iterations=args.num_iterations,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        checkpoint_dir=args.checkpoint_dir,
    )
    print("Training finished.")
