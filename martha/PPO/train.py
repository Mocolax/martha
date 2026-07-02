import argparse
import csv
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __package__ in {None, ""}:
    from buffer import RolloutBuffer
    from logic import PPOLogic
    from network import ActorCritic
else:
    from .buffer import RolloutBuffer
    from .logic import PPOLogic
    from .network import ActorCritic

from martha_env import MarthaEnv


METRIC_FIELDS = [
    "episode",
    "episode_reward",
    "episode_length",
    "terminated",
    "truncated",
    "reached_goal",
    "collision",
    "out_of_bounds",
    "updates",
    "loss",
    "actor_loss",
    "critic_loss",
    "entropy",
    "eval_mean_reward",
    "eval_success_rate",
    "best_eval_mean_reward",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena PPO continuo en MarthaEnv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--map-mode", choices=["predefined", "random"], default="predefined")
    parser.add_argument("--map-index", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--runs-dir", type=Path, default=PACKAGE_DIR / "runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--render-eval", action="store_true")
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidio --device cuda, pero CUDA no esta disponible.")
    return torch.device(device_arg)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(args):
    run_name = args.run_name
    if not run_name:
        run_name = "ppo_martha_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_metric(metrics_path, row):
    file_exists = metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, network, ppo, args, episode, best_eval_mean_reward):
    checkpoint = {
        "episode": episode,
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": ppo.optimizer.state_dict(),
        "best_eval_mean_reward": best_eval_mean_reward,
        "config": vars(args),
    }
    torch.save(checkpoint, path)


def assert_finite(name, value):
    if torch.is_tensor(value):
        is_ok = torch.isfinite(value).all().item()
    else:
        value_array = np.asarray(value, dtype=np.float32)
        is_ok = np.isfinite(value_array).all()
    if not is_ok:
        raise FloatingPointError(f"{name} contiene NaN o infinito.")


def evaluate_policy(network, args, device):
    render_mode = "human" if args.render_eval else None
    env = MarthaEnv(
        action_mode="continuous",
        render_mode=render_mode,
        map_mode=args.map_mode,
        map_index=args.map_index,
    )

    rewards = []
    successes = []
    network.eval()
    try:
        for eval_episode in range(args.eval_episodes):
            reset_seed = args.seed + 100000 + eval_episode
            observation, _ = env.reset(seed=reset_seed)
            total_reward = 0.0
            reached_goal = False

            for _ in range(args.max_steps):
                action, _, _ = network.get_action(observation, deterministic=True)
                action_np = action.numpy().astype(np.float32)
                assert_finite("eval_action", action_np)

                observation, reward, terminated, truncated, info = env.step(action_np)
                total_reward += float(reward)
                reached_goal = bool(info.get("reached_goal", False))

                if terminated or truncated:
                    break

            rewards.append(total_reward)
            successes.append(float(reached_goal))
    finally:
        env.close()
        network.train()

    return {
        "eval_mean_reward": float(np.mean(rewards)) if rewards else math.nan,
        "eval_success_rate": float(np.mean(successes)) if successes else math.nan,
    }


def build_agent(env, args, device):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    network = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    ppo = PPOLogic(
        network=network,
        lr=args.lr,
        eps=args.eps,
        gamma=args.gamma,
        lam=args.lam,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
    )
    return network, ppo


def train(args):
    set_seed(args.seed)
    device = choose_device(args.device)
    run_dir = make_run_dir(args)
    metrics_path = run_dir / "metrics.csv"
    last_model_path = run_dir / "last_model.pt"
    best_model_path = run_dir / "best_model.pt"

    env = MarthaEnv(
        action_mode="continuous",
        render_mode=None,
        map_mode=args.map_mode,
        map_index=args.map_index,
    )
    network, ppo = build_agent(env, args, device)
    buffer = RolloutBuffer()

    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    print(f"Observation shape: {env.observation_space.shape}")
    print(f"Action shape: {env.action_space.shape}")

    best_eval_mean_reward = -math.inf
    last_update_stats = {
        "loss": 0.0,
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "updates": 0,
    }
    last_observation = None
    last_done = True

    try:
        for episode in range(1, args.episodes + 1):
            reset_seed = args.seed if episode == 1 else None
            observation, _ = env.reset(seed=reset_seed)
            episode_reward = 0.0
            episode_length = 0
            terminated = False
            truncated = False
            info = {}

            for _ in range(args.max_steps):
                action, logprob, value = network.get_action(observation, deterministic=False)
                action_np = action.numpy().astype(np.float32)
                assert_finite("action", action_np)

                next_observation, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated

                buffer.store(
                    state=observation,
                    action=action_np,
                    logprob=logprob,
                    reward=reward,
                    done=done,
                    value=value,
                )

                episode_reward += float(reward)
                episode_length += 1
                observation = next_observation
                last_observation = observation
                last_done = done

                if len(buffer) >= args.rollout_steps:
                    last_value = None if done else network.get_value(observation)
                    last_update_stats = ppo.train_buffer(buffer, last_value=last_value)
                    for stat_name, stat_value in last_update_stats.items():
                        assert_finite(stat_name, stat_value)

                if done:
                    break

            eval_stats = {
                "eval_mean_reward": math.nan,
                "eval_success_rate": math.nan,
            }
            should_evaluate = args.eval_every > 0 and episode % args.eval_every == 0
            if should_evaluate:
                eval_stats = evaluate_policy(network, args, device)
                if eval_stats["eval_mean_reward"] > best_eval_mean_reward:
                    best_eval_mean_reward = eval_stats["eval_mean_reward"]
                    save_checkpoint(
                        best_model_path,
                        network,
                        ppo,
                        args,
                        episode,
                        best_eval_mean_reward,
                    )

            save_checkpoint(
                last_model_path,
                network,
                ppo,
                args,
                episode,
                best_eval_mean_reward,
            )

            metric_row = {
                "episode": episode,
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "terminated": int(terminated),
                "truncated": int(truncated),
                "reached_goal": int(bool(info.get("reached_goal", False))),
                "collision": int(bool(info.get("collision", False))),
                "out_of_bounds": int(bool(info.get("out_of_bounds", False))),
                "updates": last_update_stats["updates"],
                "loss": last_update_stats["loss"],
                "actor_loss": last_update_stats["actor_loss"],
                "critic_loss": last_update_stats["critic_loss"],
                "entropy": last_update_stats["entropy"],
                "eval_mean_reward": eval_stats["eval_mean_reward"],
                "eval_success_rate": eval_stats["eval_success_rate"],
                "best_eval_mean_reward": best_eval_mean_reward,
            }
            write_metric(metrics_path, metric_row)

            eval_text = ""
            if should_evaluate:
                eval_text = (
                    f" | eval={eval_stats['eval_mean_reward']:.2f}"
                    f" | success={eval_stats['eval_success_rate']:.2f}"
                )
            print(
                f"episode={episode:5d}"
                f" | reward={episode_reward:8.2f}"
                f" | len={episode_length:3d}"
                f" | updates={last_update_stats['updates']:3d}"
                f" | actor={last_update_stats['actor_loss']:.4f}"
                f" | critic={last_update_stats['critic_loss']:.4f}"
                f"{eval_text}"
            )

        if len(buffer) > 0:
            final_last_value = None if last_done else network.get_value(last_observation)
            last_update_stats = ppo.train_buffer(buffer, last_value=final_last_value)
            for stat_name, stat_value in last_update_stats.items():
                assert_finite(stat_name, stat_value)
            save_checkpoint(
                last_model_path,
                network,
                ppo,
                args,
                args.episodes,
                best_eval_mean_reward,
            )

        if not math.isfinite(best_eval_mean_reward):
            eval_stats = evaluate_policy(network, args, device)
            best_eval_mean_reward = eval_stats["eval_mean_reward"]
            save_checkpoint(
                best_model_path,
                network,
                ppo,
                args,
                args.episodes,
                best_eval_mean_reward,
            )
            save_checkpoint(
                last_model_path,
                network,
                ppo,
                args,
                args.episodes,
                best_eval_mean_reward,
            )

    finally:
        env.close()

    print("Training finished.")
    print(f"Metrics: {metrics_path}")
    print(f"Last model: {last_model_path}")
    print(f"Best model: {best_model_path}")
    return network, ppo


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
