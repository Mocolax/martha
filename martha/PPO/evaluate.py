import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __package__ in {None, ""}:
    from network import ActorCritic
else:
    from .network import ActorCritic

from martha_env import MarthaEnv


RESULT_FIELDS = [
    "map_index",
    "episode",
    "reward",
    "steps",
    "terminated",
    "truncated",
    "reached_goal",
    "collision",
    "out_of_bounds",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalua un checkpoint PPO continuo en los mapas predefinidos de MarthaEnv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Ruta a best_model.pt o last_model.pt.",
    )
    parser.add_argument("--episodes-per-map", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--csv", type=Path, default=None, help="Ruta opcional para guardar resultados CSV.")
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidio --device cuda, pero CUDA no esta disponible.")
    return torch.device(device_arg)


def load_policy(checkpoint_path, device):
    checkpoint_path = checkpoint_path.resolve()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    probe_env = MarthaEnv(action_mode="continuous", render_mode=None, map_mode="predefined")
    try:
        state_dim = probe_env.observation_space.shape[0]
        action_dim = probe_env.action_space.shape[0]
        hidden_dim = checkpoint.get("config", {}).get("hidden_dim", 256)
        network = ActorCritic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.eval()
        map_count = len(probe_env.predefined_maps)
    finally:
        probe_env.close()

    return network, checkpoint, map_count


def evaluate_episode(network, map_index, args):
    render_mode = "human" if args.render else None
    env = MarthaEnv(
        action_mode="continuous",
        render_mode=render_mode,
        map_mode="random",
        map_index=map_index,
    )

    try:
        observation, _ = env.reset(seed=args.seed + map_index)
        total_reward = 0.0
        info = {}
        terminated = False
        truncated = False
        steps = 0

        for steps in range(1, args.max_steps + 1):
            action, _, _ = network.get_action(observation, deterministic=True)
            action = action.numpy().astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                break

        return {
            "reward": total_reward,
            "steps": steps,
            "terminated": int(terminated),
            "truncated": int(truncated),
            "reached_goal": int(bool(info.get("reached_goal", False))),
            "collision": int(bool(info.get("collision", False))),
            "out_of_bounds": int(bool(info.get("out_of_bounds", False))),
        }
    finally:
        env.close()


def write_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, checkpoint_path):
    print(f"\nCheckpoint: {checkpoint_path}")
    print("map | episodes | mean_reward | success | collision | out_of_bounds | truncated")
    print("----|----------|-------------|---------|-----------|---------------|----------")

    all_rewards = []
    all_successes = []
    for map_index in sorted({row["map_index"] for row in rows}):
        map_rows = [row for row in rows if row["map_index"] == map_index]
        rewards = [row["reward"] for row in map_rows]
        successes = [row["reached_goal"] for row in map_rows]
        collisions = [row["collision"] for row in map_rows]
        out_of_bounds = [row["out_of_bounds"] for row in map_rows]
        truncated = [row["truncated"] for row in map_rows]
        all_rewards.extend(rewards)
        all_successes.extend(successes)

        print(
            f"{map_index:>3} |"
            f" {len(map_rows):>8} |"
            f" {np.mean(rewards):>11.2f} |"
            f" {np.mean(successes):>7.2f} |"
            f" {np.mean(collisions):>9.2f} |"
            f" {np.mean(out_of_bounds):>13.2f} |"
            f" {np.mean(truncated):>8.2f}"
        )

    print("----|----------|-------------|---------|-----------|---------------|----------")
    print(
        f"all |"
        f" {len(rows):>8} |"
        f" {np.mean(all_rewards):>11.2f} |"
        f" {np.mean(all_successes):>7.2f} |"
        f" {'':>9} |"
        f" {'':>13} |"
        f" {'':>8}"
    )


def main():
    args = parse_args()
    device = choose_device(args.device)
    network, checkpoint, map_count = load_policy(args.checkpoint, device)

    print(f"Device: {device}")
    print(f"Checkpoint episode: {checkpoint.get('episode', 'unknown')}")
    print(f"Best eval mean reward: {checkpoint.get('best_eval_mean_reward', 'unknown')}")
    print(f"Testing {map_count} predefined maps...")

    rows = []
    with torch.no_grad():
        for map_index in range(map_count):
            for episode in range(1, args.episodes_per_map + 1):
                result = evaluate_episode(network, map_index, args)
                row = {
                    "map_index": map_index,
                    "episode": episode,
                    **result,
                }
                rows.append(row)
                print(
                    f"map={map_index:02d}"
                    f" episode={episode:02d}"
                    f" reward={row['reward']:8.2f}"
                    f" steps={row['steps']:3d}"
                    f" goal={row['reached_goal']}"
                    f" collision={row['collision']}"
                    f" oob={row['out_of_bounds']}"
                    f" truncated={row['truncated']}"
                )

    print_summary(rows, args.checkpoint)

    if args.csv is not None:
        write_csv(args.csv, rows)
        print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
