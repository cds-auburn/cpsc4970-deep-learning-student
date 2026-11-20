"""Run one fixed forecast and save its numerical results and lecture plots."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import StrMethodFormatter

from electricity_model import MODULE, load_example, load_pipeline

HISTORY, MODEL = "#16465c", "#a34700"


def plot_forecast(times, history, predictions=None, actual=None, band=False, days=7):
    """Plot the last `days` of history, then whichever forecast layers are given.

    Every figure has the same size and vertical scale, so slides can stack them.
    """
    first = 168 - 24 * days
    fig, ax = plt.subplots(figsize=(12.5, 3.75))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.9, bottom=0.2)  # Fixed, so frames align.
    ax.plot(times[first:168], history[0, first:].numpy(), color=HISTORY, linewidth=2.5, label="History")
    ax.axvline(times[168], color="#333333", linestyle=":", linewidth=1.5)
    ax.text(times[168], 18600, " Forecast starts", ha="left", va="bottom", fontsize=14)
    if band:
        ax.fill_between(times[168:], predictions[0, 0].numpy(), predictions[0, 8].numpy(),
                        color=MODEL, alpha=0.2, linewidth=0, label="Model's 0.1–0.9 quantiles")
    if predictions is not None:
        ax.plot(times[168:], predictions[0, 4].numpy(), color=MODEL, linestyle="--", linewidth=3,
                label="Median forecast")
    if actual is not None:
        ax.plot(times[168:], actual.numpy(), color=HISTORY, linewidth=2, marker="o", markersize=5,
                label="Observed later")
    ax.set(xlim=(times[first], times[-1] + (times[1] - times[0]) / 2), ylim=(18000, 47000), ylabel="Demand (MW)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    if days <= 3:
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", which="minor", labelsize=13, labelcolor="#444444")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(labelsize=14)
    ax.yaxis.label.set_size(15)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    # The legend sits above the axes: inside, it would cover a nightly minimum.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=4, fontsize=14, frameon=False,
              borderaxespad=0.1, handlelength=2.5)
    fig.text(0.985, 0.02, "January 2024, UTC. Data: NESO. Supported by National Energy SO Open Data.",
             ha="right", va="bottom", fontsize=12, color="#444444")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=MODULE / "artifacts/electricity")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")  # Save files; a notebook importing plot_forecast keeps its own.
    plt.rcParams["svg.hashsalt"] = "M2-080"  # Reproducible SVG element IDs.
    torch.set_num_threads(2)
    times, history, actual = load_example()
    pipeline = load_pipeline()
    forecasts = pipeline.predict(history, prediction_length=24)
    assert forecasts.shape == (1, 9, 24)
    assert torch.isfinite(forecasts).all()
    results = {
        "model": "amazon/chronos-bolt-tiny", "input_shape": list(history.shape),
        "output_shape": list(forecasts.shape), "quantiles": pipeline.quantiles,
        "forecast_start": times[168].isoformat(), "forecast_end": times[-1].isoformat(),
        "forecast_mw": forecasts[0].tolist(), "actual_mw": actual.tolist(),
        "last_observation_mw": history[0, -1].item(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in
               ("torch", "chronos-forecasting", "transformers", "numpy", "pandas", "matplotlib")}},
    }
    (args.output_dir / "example-results.json").write_text(json.dumps(results, indent=2) + "\n")
    # Week-long views show the forecast beside every known day. The comparisons
    # keep one known day, so the forecast day fills half of the width.
    for name, pred, obs, band, days in [
        ("history", None, None, False, 7), ("forecast", forecasts, None, False, 7),
        ("forecast-day", forecasts, None, False, 1),
        ("observed", forecasts, actual, False, 1), ("interval", forecasts, actual, True, 1),
    ]:
        fig = plot_forecast(times, history, pred, obs, band, days)
        fig.savefig(args.output_dir / f"M2-080-{name}.svg")
        plt.close(fig)
    print("Input:", tuple(history.shape), "Output:", tuple(forecasts.shape))
    print("First three median forecasts:", forecasts[0, 4, :3].tolist())
    print("First three observed values:", actual[:3].tolist())
    print("Last known:", history[0, -1].item())
    print("Saved to", args.output_dir)


if __name__ == "__main__":
    main()
