"""Generate readable learning reports from a Martha PPO metrics CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


# Edit this value to change the smoothing used by every generated report.
REPORT_WINDOW = 50
DEFAULT_RUNS_DIRECTORY = (
    Path.home() / "ros2_ws/src/martha/martha/PPO/ppo_runs"
)

REWARD_COMPONENTS = (
    ("reward_distance", "Distancia"),
    ("reward_orientation", "Orientación"),
    ("reward_shortest_distance", "Récord de cercanía"),
    ("reward_laser", "Láser"),
    ("reward_wiggle", "Zigzag"),
    ("reward_terminal", "Terminal"),
)


def _load_metrics(path: Path) -> dict[str, np.ndarray]:
    """Load, sort and deduplicate numeric episode metrics."""
    if not path.is_file():
        raise FileNotFoundError(f"metrics CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"metrics CSV has no episode rows: {path}")

    by_episode = {}
    for row in rows:
        try:
            episode = int(float(row["episode"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("metrics CSV contains an invalid episode") from exc
        by_episode[episode] = row
    ordered = [by_episode[index] for index in sorted(by_episode)]
    columns = set().union(*(row.keys() for row in ordered))
    metrics: dict[str, np.ndarray] = {
        "episode": np.asarray(sorted(by_episode), dtype=np.float64)
    }
    for name in columns - {"episode"}:
        values = []
        for row in ordered:
            try:
                values.append(float(row.get(name, "nan")))
            except (TypeError, ValueError):
                values.append(np.nan)
        metrics[name] = np.asarray(values, dtype=np.float64)
    return metrics


def _values(metrics: dict[str, np.ndarray], name: str) -> np.ndarray:
    """Return one metric from the current CSV schema."""
    try:
        return metrics[name]
    except KeyError as exc:
        if name.startswith("reward_"):
            # Component names changed with the paper reward.  Older runs can
            # still produce their learning report, without a component plot.
            return np.full(metrics["episode"].shape, np.nan, dtype=np.float64)
        raise ValueError(
            f"metrics CSV is missing current field {name!r}"
        ) from exc


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    """Calculate a trailing finite-value mean with partial initial windows."""
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    kernel = np.ones(max(1, int(window)), dtype=np.float64)
    sums = np.convolve(clean, kernel, mode="full")[: len(values)]
    counts = np.convolve(finite.astype(np.float64), kernel, mode="full")[
        : len(values)
    ]
    return np.divide(
        sums,
        counts,
        out=np.full(values.shape, np.nan, dtype=np.float64),
        where=counts > 0.0,
    )


def _plot_smoothed(
    axis,
    episodes: np.ndarray,
    values: np.ndarray,
    label: str,
    window: int,
    *,
    raw: bool = False,
) -> None:
    if not np.isfinite(values).any():
        return
    if raw:
        axis.plot(episodes, values, alpha=0.16, linewidth=0.7)
    axis.plot(episodes, _rolling(values, window), linewidth=2.0, label=label)


def _finite_points(
    episodes: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(values)
    return episodes[mask], values[mask]


def _window_mean(values: np.ndarray, window: int, *, first: bool) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    sample = finite[:window] if first else finite[-window:]
    return float(np.mean(sample))


def _format_change(first: float, last: float, percent: bool = False) -> str:
    if not np.isfinite(first) or not np.isfinite(last):
        return "sin datos"
    multiplier = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return (
        f"{first * multiplier:.2f}{suffix} -> "
        f"{last * multiplier:.2f}{suffix}"
    )


def _write_summary(
    path: Path,
    metrics: dict[str, np.ndarray],
    window: int,
) -> None:
    """Write numerical comparisons and cautious diagnostic suggestions."""
    effective = min(window, len(metrics["episode"]))
    tracked = (
        ("episode_reward", "Recompensa", False),
        ("reached_goal", "Tasa de éxito", True),
        ("collision", "Tasa de colisión", True),
        ("out_of_bounds", "Fuera de límites", True),
        ("truncated", "Tasa de truncamiento", True),
        ("spl", "SPL", False),
        ("episode_length", "Duración", False),
    )
    means = {
        name: (
            _window_mean(_values(metrics, name), effective, first=True),
            _window_mean(_values(metrics, name), effective, first=False),
        )
        for name, _, _ in tracked
    }
    lines = [
        "INFORME DE ENTRENAMIENTO PPO MARTHA",
        "====================================",
        f"Episodios registrados: {len(metrics['episode'])}",
        f"Ventana comparada: {effective} episodios",
        "",
        "Primeros -> últimos episodios",
    ]
    for name, label, percent in tracked:
        lines.append(
            f"- {label}: {_format_change(*means[name], percent=percent)}"
        )

    lines.extend(("", "Componentes medios de recompensa (última ventana)"))
    for name, label in REWARD_COMPONENTS:
        value = _window_mean(_values(metrics, name), effective, first=False)
        lines.append(
            f"- {label}: {value:.4f}" if np.isfinite(value) else f"- {label}: sin datos"
        )

    recommendations = []
    success = means["reached_goal"][1]
    collision = means["collision"][1]
    out_of_bounds = means["out_of_bounds"][1]
    truncated = means["truncated"][1]
    explained = _window_mean(
        _values(metrics, "explained_variance"), effective, first=False
    )
    approx_kl = _window_mean(
        _values(metrics, "approx_kl"), effective, first=False
    )
    clip_fraction = _window_mean(
        _values(metrics, "clip_fraction"), effective, first=False
    )
    if np.isfinite(collision) and collision > 0.50:
        recommendations.append(
            "La colisión supera 50%: revisa la huella/márgenes y considera "
            "más señal de clearance o velocidades menores."
        )
    if np.isfinite(truncated) and truncated > 0.50:
        recommendations.append(
            "Más de 50% llega al límite de steps: verifica que la recompensa "
            "de progreso domine el costo por paso y que las metas sean alcanzables."
        )
    if np.isfinite(out_of_bounds) and out_of_bounds > 0.05:
        recommendations.append(
            "Hay salidas del mapa: revisa la geometría, los resets y el margen "
            "de los límites antes de cambiar PPO."
        )
    if np.isfinite(success) and success < 0.05:
        recommendations.append(
            "Éxito menor a 5%: usa metas más cortas/currículo antes de ajustar "
            "finamente penalizaciones pequeñas."
        )
    if np.isfinite(explained) and explained < 0.0:
        recommendations.append(
            "Explained variance negativa: el crítico no está modelando los "
            "retornos; revisa reward_scale, learning rate y estabilidad."
        )
    if (
        np.isfinite(approx_kl)
        and np.isfinite(clip_fraction)
        and approx_kl < 1e-4
        and clip_fraction < 0.01
    ):
        recommendations.append(
            "KL y clip fraction casi nulos: las actualizaciones pueden ser "
            "demasiado débiles o la política puede haberse estancado."
        )
    if (
        np.isfinite(approx_kl)
        and np.isfinite(clip_fraction)
        and (approx_kl > 0.05 or clip_fraction > 0.30)
    ):
        recommendations.append(
            "KL/clip fraction elevados: considera menor learning rate o menos "
            "épocas PPO por rollout."
        )
    if not recommendations:
        recommendations.append(
            "No se activó ninguna alerta simple; compara especialmente la "
            "evaluación determinista y las tendencias, no un episodio aislado."
        )
    lines.extend(("", "Indicadores PPO (última ventana)"))
    lines.append(f"- Explained variance: {explained:.4f}")
    lines.append(f"- Approx KL: {approx_kl:.6f}")
    lines.append(f"- Clip fraction: {clip_fraction:.4f}")
    lines.extend(("", "Sugerencias automáticas (heurísticas)"))
    lines.extend(f"- {recommendation}" for recommendation in recommendations)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_training_report(
    metrics_path: str | Path,
    window: int = REPORT_WINDOW,
) -> tuple[Path, ...]:
    """Create learning, PPO-diagnostic and text reports beside metrics.csv."""
    if window <= 0:
        raise ValueError("report smoothing window must be positive")
    metrics_path = Path(metrics_path).expanduser().resolve()
    metrics = _load_metrics(metrics_path)
    episodes = metrics["episode"]
    run_directory = metrics_path.parent

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")
    figure, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    figure.suptitle(
        f"Aprendizaje PPO Martha — media móvil de {window} episodios",
        fontsize=16,
    )

    reward = _values(metrics, "episode_reward")
    _plot_smoothed(axes[0, 0], episodes, reward, "Recompensa", window, raw=True)
    axes[0, 0].set_title("Recompensa de entrenamiento")
    axes[0, 0].set_ylabel("Recompensa original")

    for name, label in (
        ("reached_goal", "Éxito"),
        ("collision", "Colisión"),
        ("truncated", "Truncado"),
        ("out_of_bounds", "Fuera del mapa"),
    ):
        _plot_smoothed(axes[0, 1], episodes, _values(metrics, name), label, window)
    axes[0, 1].set_title("Resultados por episodio")
    axes[0, 1].set_ylabel("Tasa")
    axes[0, 1].set_ylim(-0.03, 1.03)

    _plot_smoothed(
        axes[1, 0], episodes, _values(metrics, "episode_length"),
        "Steps", window, raw=True,
    )
    axes[1, 0].set_title("Duración de los episodios")
    axes[1, 0].set_ylabel("Steps")

    _plot_smoothed(
        axes[1, 1], episodes, _values(metrics, "spl"), "SPL", window,
    )
    axes[1, 1].set_title("Eficiencia de ruta (SPL)")
    axes[1, 1].set_ylabel("SPL")
    axes[1, 1].set_ylim(-0.03, 1.03)

    has_components = False
    for name, label in REWARD_COMPONENTS:
        values = _values(metrics, name)
        if np.isfinite(values).any():
            has_components = True
            _plot_smoothed(axes[2, 0], episodes, values, label, window)
    axes[2, 0].set_title("Contribución de cada término de recompensa")
    axes[2, 0].set_ylabel("Suma por episodio")
    if not has_components:
        axes[2, 0].text(
            0.5, 0.5, "Este run es anterior al log por componentes",
            ha="center", va="center", transform=axes[2, 0].transAxes,
        )

    evaluation_found = False
    for name, label in (
        ("eval_success_rate", "Éxito eval"),
        ("eval_collision_rate", "Colisión eval"),
        ("eval_mean_spl", "SPL eval"),
    ):
        x_values, y_values = _finite_points(episodes, _values(metrics, name))
        if y_values.size:
            evaluation_found = True
            axes[2, 1].plot(x_values, y_values, marker="o", label=label)
    axes[2, 1].set_title("Evaluación determinista")
    axes[2, 1].set_ylabel("Tasa / SPL")
    axes[2, 1].set_ylim(-0.03, 1.03)
    if not evaluation_found:
        axes[2, 1].text(
            0.5, 0.5, "Aún no hay evaluaciones periódicas",
            ha="center", va="center", transform=axes[2, 1].transAxes,
        )

    for axis in axes.flat:
        axis.set_xlabel("Episodio")
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best")
    learning_path = run_directory / "learning_report.png"
    figure.savefig(learning_path, dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle("Diagnóstico interno de PPO", fontsize=16)
    for name, label in (
        ("actor_loss", "Actor loss"),
        ("critic_loss", "Critic loss"),
        ("loss", "Loss total"),
    ):
        _plot_smoothed(axes[0, 0], episodes, _values(metrics, name), label, window)
    axes[0, 0].set_title("Pérdidas")

    for name, label in (("entropy", "Entropía"), ("policy_std", "Policy std")):
        _plot_smoothed(axes[0, 1], episodes, _values(metrics, name), label, window)
    axes[0, 1].set_title("Exploración")

    for name, label in (
        ("approx_kl", "Approx KL"),
        ("clip_fraction", "Clip fraction"),
    ):
        _plot_smoothed(axes[1, 0], episodes, _values(metrics, name), label, window)
    axes[1, 0].set_title("Magnitud de las actualizaciones")

    for name, label in (
        ("explained_variance", "Explained variance"),
        ("actor_inactive_relu", "ReLU inactivas actor"),
        ("critic_inactive_relu", "ReLU inactivas crítico"),
    ):
        _plot_smoothed(axes[1, 1], episodes, _values(metrics, name), label, window)
    axes[1, 1].set_title("Crítico y activaciones")
    for axis in axes.flat:
        axis.set_xlabel("Episodio")
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best")
    diagnostics_path = run_directory / "ppo_diagnostics.png"
    figure.savefig(diagnostics_path, dpi=150)
    plt.close(figure)

    summary_path = run_directory / "training_summary.txt"
    _write_summary(summary_path, metrics, window)
    return learning_path, diagnostics_path, summary_path


def _latest_metrics(runs_directory: Path) -> Path:
    candidates = list(runs_directory.glob("*/metrics.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"no metrics.csv files were found under {runs_directory}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main(argv: Iterable[str] | None = None) -> None:
    """Regenerate a report for one run, or the most recently modified run."""
    parser = argparse.ArgumentParser(
        description="Generate Martha PPO learning plots from metrics.csv."
    )
    parser.add_argument(
        "run",
        nargs="?",
        type=Path,
        help="Run directory or metrics.csv; defaults to the latest run.",
    )
    args = parser.parse_args(argv)
    target = args.run
    if target is None:
        metrics_path = _latest_metrics(DEFAULT_RUNS_DIRECTORY)
    else:
        target = target.expanduser().resolve()
        metrics_path = target / "metrics.csv" if target.is_dir() else target
    for path in generate_training_report(metrics_path):
        print(path)


if __name__ == "__main__":
    main()
