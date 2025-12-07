import argparse
import json
import math
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mlx.core as mx

from env import Env
from train import Actor

ENV_DEFAULTS = {
    "m": 2.0,
    "l": 1.0,
    "g": 9.81,
    "dt": 0.02,
    "max_torque": 1.0,
    "theta_threshold": math.pi / 2,
    "max_theta_dot": 10.0,
    "init_theta_range": math.pi,
    "init_theta_dot_range": 1.0,
}


def rollout(actor: Actor, env: Env, max_steps: int) -> dict[str, list[float]]:
    obs = env.reset()
    thetas: list[float] = []
    theta_dots: list[float] = []

    for _ in range(max_steps):
        mean, _ = actor(obs)
        action = mean
        obs, _, done, info = env.step(action)

        thetas.append(float(info["theta"].item()))
        theta_dots.append(float(info["theta_dot"].item()))

        if done:
            break

    return {"theta": thetas, "theta_dot": theta_dots}


def build_animation(history: dict[str, list[float]], env: Env):
    l = float(env.l.item())
    dt = float(env.dt.item())
    thetas = history["theta"]
    theta_dots = history["theta_dot"]
    times = [i * dt for i in range(len(thetas))]

    xs = [l * math.sin(theta) for theta in thetas]
    ys = [l * math.cos(theta) for theta in thetas]

    fig, (ax_pend, ax_theta) = plt.subplots(1, 2, figsize=(10, 4))
    line, = ax_pend.plot([], [], "o-", lw=4)
    ax_pend.set_xlim(-l - 0.2, l + 0.2)
    ax_pend.set_ylim(-l - 0.2, l + 0.2)
    ax_pend.set_aspect("equal", "box")
    ax_pend.set_title("Inverted Pendulum")
    ax_pend.grid(True)

    theta_line, = ax_theta.plot([], [], label="theta (rad)")
    theta_dot_line, = ax_theta.plot([], [], label="theta_dot (rad/s)")
    ax_theta.set_xlim(0, max(times) if times else 1.0)
    y_min = min(thetas + theta_dots, default=-1.0)
    y_max = max(thetas + theta_dots, default=1.0)
    ax_theta.set_ylim(y_min * 1.1, y_max * 1.1 if y_max != 0 else 1.0)
    ax_theta.set_xlabel("Time (s)")
    ax_theta.legend()
    ax_theta.grid(True)

    def init():
        line.set_data([], [])
        theta_line.set_data([], [])
        theta_dot_line.set_data([], [])
        return line, theta_line, theta_dot_line

    def update(frame):
        line.set_data([0.0, xs[frame]], [0.0, ys[frame]])
        theta_line.set_data(times[: frame + 1], thetas[: frame + 1])
        theta_dot_line.set_data(times[: frame + 1], theta_dots[: frame + 1])
        return line, theta_line, theta_dot_line

    pendulum_animation = animation.FuncAnimation(
        fig,
        update,
        frames=len(thetas),
        init_func=init,
        interval=dt * 1000.0,
        blit=True,
        repeat=False,
    )
    return fig, pendulum_animation


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained inverted pendulum policy")
    parser.add_argument(
        "--actor_weights",
        type=Path,
        default=Path("training/inv_pendulum/checkpoints/actor_latest.npz"),
        help="Path to the saved actor weights.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a saved training config. Defaults to actor directory.",
    )
    parser.add_argument("--max_steps", type=int, default=1000, help="Number of frames to simulate.")
    parser.add_argument("--m", type=float, default=None, help="Override pendulum mass.")
    parser.add_argument("--l", type=float, default=None, help="Override pendulum length.")
    parser.add_argument(
        "--max_torque",
        type=float,
        default=None,
        help="Maximum torque magnitude to keep consistent with training.",
    )
    parser.add_argument(
        "--init_theta_range",
        type=float,
        default=None,
        help="Initial theta sampled uniformly from [-range, range].",
    )
    parser.add_argument(
        "--init_theta_dot_range",
        type=float,
        default=None,
        help="Initial angular velocity sampled uniformly from [-range, range].",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=None,
        help="Optional path to save the animation (e.g., mp4 or gif).",
    )
    args = parser.parse_args()

    if not args.actor_weights.exists():
        raise FileNotFoundError(f"Could not find actor weights: {args.actor_weights}")

    config_path = args.config or args.actor_weights.parent / "config.json"
    config_data = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
    elif args.config is not None:
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    env_config = {key: ENV_DEFAULTS[key] for key in ENV_DEFAULTS}
    env_config.update(config_data.get("env", {}))
    overrides = {
        "m": args.m,
        "l": args.l,
        "max_torque": args.max_torque,
        "init_theta_range": args.init_theta_range,
        "init_theta_dot_range": args.init_theta_dot_range,
    }
    for key, value in overrides.items():
        if value is not None:
            env_config[key] = value

    # Ensure visualization starts within upright hemisphere to avoid upside-down initial states.
    upright_limit = (math.pi / 2) - 1e-3
    env_config["init_theta_range"] = min(env_config["init_theta_range"], upright_limit)

    env = Env(
        m=env_config["m"],
        l=env_config["l"],
        g=env_config["g"],
        dt=env_config["dt"],
        max_torque=env_config["max_torque"],
        theta_threshold=env_config["theta_threshold"],
        max_theta_dot=env_config["max_theta_dot"],
        init_theta_range=env_config["init_theta_range"],
        init_theta_dot_range=env_config["init_theta_dot_range"],
    )
    obs_dim = env.reset().shape[-1]
    action_dim = 1
    actor = Actor(obs_dim, action_dim)
    actor.load_weights(str(args.actor_weights))
    mx.eval(actor.parameters())

    history = rollout(actor, env, args.max_steps)
    if not history["theta"]:
        raise RuntimeError("Rollout finished immediately; cannot visualize.")

    fig, pendulum_animation = build_animation(history, env)

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        pendulum_animation.save(args.save_path, writer="ffmpeg", fps=int(1.0 / float(env.dt.item())))
        print(f"Saved animation to {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
