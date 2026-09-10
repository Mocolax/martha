"""Demo en vivo: MARTHA (Best-10m) navegando en el laboratorio.

Requiere simulation.launch.py corriendo con world:=lab.world (GUI encendida).
Recorre en bucle rutas medias que la politica resuelve, hasta Ctrl+C.
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path
import torch

REPO = Path("/home/ros/ros2_ws/src/martha")
sys.path.insert(0, str(REPO))
from martha.PPO import world_map, training_layout
world_map.TRAINING_WORLD_NAMES = ("lab",)
training_layout.WORLD_ORIGINS = {"lab": (0.0, 0.0)}
training_layout.TRAINING_POINT_COLORS = {"lab": (1.0, 0.55, 0.0, 1.0)}

from martha.PPO.evaluate import EvaluationDefaults, make_environment
from martha.PPO.checkpoint import load_policy, choose_device
from martha.PPO.evaluation_core import episode_path_state

POINTS = {1: (-1.60, 2.60), 2: (2.90, 3.40), 3: (-1.80, 0.60),
          4: (0.90, -0.60), 5: (-0.60, -3.15), 6: (2.40, -2.40)}
# Rutas variadas que la politica resuelve bien (corta, media, diagonal larga).
ROUTES = [(3, 1), (1, 6), (6, 3), (4, 3), (6, 4), (1, 3), (2, 4)]
CKPT = REPO / "martha/PPO/ppo_runs/Best-10m/best_model.pt"
MAX_STEPS = 800
DETERMINISTIC = True  # ponlo en False para dejar exploracion (mas exito con reintentos)

def main():
    device = choose_device("auto")
    network, checkpoint, limits = load_policy(CKPT, device)
    args = EvaluationDefaults(checkpoint=CKPT, max_steps=MAX_STEPS, backend="gazebo")
    env = make_environment(args, checkpoint, limits)
    print(f"\n>>> DEMO MARTHA — checkpoint Best-10m (ep {checkpoint.get('episode')}) <<<\n", flush=True)
    try:
        with torch.no_grad():
            for k in itertools.count():
                i, j = ROUTES[k % len(ROUTES)]
                obs, info = env.reset(options={"start": POINTS[i], "goal": POINTS[j]})
                shortest = episode_path_state(info)[0]
                print(f"[ruta {i}->{j}]  distancia ~{shortest:.1f} m  ... navegando", flush=True)
                rec = network.initial_recurrent_state(1); first = True
                reached = collided = False
                for step in range(1, MAX_STEPS + 1):
                    action, _, _, rec = network.get_actions_recurrent(
                        obs, rec, episode_starts=[first], deterministic=DETERMINISTIC)
                    first = False
                    obs, _, term, trunc, info = env.step(action.squeeze(0).numpy().astype("float32"))
                    if term or trunc:
                        reached = bool(info.get("reached_goal")); collided = bool(info.get("collision"))
                        break
                tag = "LLEGO" if reached else ("CHOCO" if collided else "sin llegar")
                print(f"          -> {tag}  ({step} pasos)\n", flush=True)
    except KeyboardInterrupt:
        print("\n(demo detenida)")
    finally:
        env.close()

if __name__ == "__main__":
    main()
