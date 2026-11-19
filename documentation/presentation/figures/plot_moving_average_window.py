"""
Draws the illustration of the Moving Average Of The Weights section of the README (README/moving_average_window.png),
which shows how many trained weights one averaged model mixes, computed from the decays the trainer applies
(compute_update_decay of vision_training/trainers/weight_averaging.py, used by every update of
ExponentialMovingAverage): on the left, the share of the averaged weights of the last iteration held by the trained
weights of every block of iterations, with the warm-up of the trainer and with the configured decay used from the first
update; on the right, the window 1 / (1 - decay) of the update of every iteration, which the warm-up keeps below the
configured window for 9 / (1 - decay) iterations. The defaults are those of the detection benchmark launches of the
figure below it (decay 0.9998, stopped at iteration 15,000). From the repository root:

    uv run python -m documentation.presentation.figures.plot_moving_average_window \
        --output_path README/moving_average_window.png
"""

import argparse
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import List, Tuple

from vision_training.trainers.weight_averaging import compute_recent_updates_weight, compute_update_decay

# Colour of the averaged weights as the trainer computes them (the accent of the moving average figures), colour of
# the same decay used from the first update without warm-up, and the dark grey of the reference lines and their labels
TRAINER_COLOUR, CONSTANT_DECAY_COLOUR = "#1F77B4", "#9C4F96"
REFERENCE_COLOUR, TEXT_COLOUR = "#7F7F7F", "#333333"

# Width of the blocks of iterations whose shares of the average are drawn on the left
BLOCK_ITERATIONS = 100

# Smaller decay drawn as a reference line on the right, the one of the shorter runs
REFERENCE_DECAY = 0.999


def compute_update_weights(decay: float, number_updates: int, warm_up: bool) -> Tuple[List[float], float]:
    """
    Gives the share of the averaged weights held by the trained weights of every update, and the share left to the
    weights the average started from, after the given number of updates: update n multiplies everything before it by
    its decay d_n and adds its trained weights with 1 - d_n, so its trained weights keep (1 - d_n) times the product of
    the decays of the later updates. With the warm-up, d_n is the decay compute_update_decay gives the trainer; without
    it, the configured decay for every update, to show what the warm-up avoids.

    Args:
        decay (float): The configured decay of the average, strictly between 0 and 1.
        number_updates (int): The number of updates of the average, one per training iteration.
        warm_up (bool): Whether the decays follow the warm-up of the trainer.

    Returns:
        Tuple[List[float], float]: The share of every update in order, and the share of the starting weights; together
            they sum to 1.
    """
    if number_updates < 1:
        raise ValueError(f"Expected at least one update of the moving average, got {number_updates}.")
    update_decays = [compute_update_decay(decay=decay, update_index=update_index) if warm_up else decay
                     for update_index in range(number_updates)]

    # Walk back from the last update, the product of the later decays shrinking the share of every older update
    update_shares = [0.0] * number_updates
    later_decays_product = 1.0
    for update_index in range(number_updates - 1, -1, -1):
        update_shares[update_index] = (1.0 - update_decays[update_index]) * later_decays_product
        later_decays_product *= update_decays[update_index]
    return update_shares, later_decays_product


def sum_block_shares(update_shares: List[float], block_iterations: int) -> Tuple[List[int], List[float]]:
    """
    Sums the shares of the updates by blocks of iterations, so that the left panel reads as the weight of every block
    of the training in the averaged model instead of shares too small to read per update.

    Args:
        update_shares (List[float]): The share of every update, update n being iteration n + 1.
        block_iterations (int): The number of iterations of a block.

    Returns:
        Tuple[List[int], List[float]]: The last iteration of every block and the share of the block in percent.
    """
    block_ends, block_shares = [], []
    for block_start in range(0, len(update_shares), block_iterations):
        block = update_shares[block_start:block_start + block_iterations]
        block_ends.append(block_start + len(block))
        block_shares.append(100.0 * sum(block))
    return block_ends, block_shares


def compute_window_curve(decay: float, last_iteration: int) -> Tuple[List[int], List[float]]:
    """
    Gives the window 1 / (1 - d) of the decay d of the update of every iteration, the number of past iterations the
    average mostly looks back over at that point of a run, (9 + n) / 9 during the warm-up and 1 / (1 - decay) after it.

    Args:
        decay (float): The configured decay of the average.
        last_iteration (int): The last iteration drawn.

    Returns:
        Tuple[List[int], List[float]]: The iterations and their windows in iterations.
    """
    iterations = list(range(1, last_iteration + 1))
    windows = [1.0 / (1.0 - compute_update_decay(decay=decay, update_index=iteration - 1)) for iteration in iterations]
    return iterations, windows


def format_thousands(value: float, position: int) -> str:
    """
    Writes an axis tick with a thousands separator, as the iterations of the README text.

    Args:
        value (float): The tick value.
        position (int): The tick position, unused.

    Returns:
        str: The tick label.
    """
    return f"{value:,.0f}"


def draw_composition_panel(axis: plt.Axes, decay: float, last_iteration: int) -> None:
    """
    Draws the left panel: what the averaged weights of the last iteration are made of, block by block, with the warm-up
    of the trainer and with the configured decay used from the first update, the window of the last update shaded with
    the share it holds.

    Args:
        axis (plt.Axes): The axis of the panel.
        decay (float): The configured decay of the average.
        last_iteration (int): The iteration whose averaged weights are decomposed.
    """
    # Shares of the trained weights of every block, with and without the warm-up of the trainer
    trainer_shares, trainer_start_share = compute_update_weights(decay=decay, number_updates=last_iteration,
                                                                 warm_up=True)
    constant_shares, constant_start_share = compute_update_weights(decay=decay, number_updates=last_iteration,
                                                                   warm_up=False)
    block_ends, trainer_block_shares = sum_block_shares(update_shares=trainer_shares, block_iterations=BLOCK_ITERATIONS)
    _, constant_block_shares = sum_block_shares(update_shares=constant_shares, block_iterations=BLOCK_ITERATIONS)

    # Window of the decay of the last update and the share of the average its iterations hold
    last_update_decay = compute_update_decay(decay=decay, update_index=last_iteration - 1)
    last_update_window = round(1.0 / (1.0 - last_update_decay))
    configured_window = round(1.0 / (1.0 - decay))
    window_share = 100.0 * compute_recent_updates_weight(decay=decay, number_updates=last_iteration,
                                                         recent_updates=last_update_window)
    configured_window_share = 100.0 * compute_recent_updates_weight(decay=decay, number_updates=last_iteration,
                                                                    recent_updates=min(configured_window,
                                                                                       last_iteration))
    constant_configured_share = 100.0 * sum(constant_shares[-configured_window:])

    # Blocks of the trainer as a filled step curve, the constant decay as a dashed step line over it
    axis.fill_between(block_ends, trainer_block_shares, step="pre", color=TRAINER_COLOUR, alpha=0.30, linewidth=0)
    axis.step(block_ends, trainer_block_shares, where="pre", color=TRAINER_COLOUR, linewidth=2,
              label=f"with the warm-up of the trainer (last update {last_update_decay:.4f})")
    axis.step(block_ends, constant_block_shares, where="pre", color=CONSTANT_DECAY_COLOUR, linewidth=2,
              linestyle="--", label=f"decay {decay} from the first update")

    # Window of the last update shaded, with the share of the average it holds
    axis.axvspan(xmin=last_iteration - last_update_window, xmax=last_iteration, color=TRAINER_COLOUR, alpha=0.10,
                 linewidth=0, zorder=0)
    top_share = max(trainer_block_shares)
    axis.text(x=last_iteration - last_update_window - 150, y=0.80 * top_share,
              s=f"last {last_update_window:,} iterations:\n{window_share:.0f} % of the average\n"
                f"last {configured_window:,}: {configured_window_share:.0f} %",
              ha="right", va="center", fontsize=12, color=TEXT_COLOUR)

    # What the configured decay alone would keep: the last configured window and the random starting weights
    axis.text(x=0.02 * last_iteration, y=0.36 * top_share,
              s=f"decay {decay} from the first update:\nlast {configured_window:,} iterations "
                f"{constant_configured_share:.0f} %,\nrandom starting weights {100.0 * constant_start_share:.1f} %",
              ha="left", va="center", fontsize=12, color=TEXT_COLOUR)

    # Axis labels, title and legend
    axis.set_xlim(left=0, right=last_iteration)
    axis.set_ylim(bottom=0, top=1.12 * top_share)
    axis.xaxis.set_major_formatter(FuncFormatter(format_thousands))
    axis.set_xlabel(xlabel="Iteration whose trained weights entered the average", fontsize=13)
    axis.set_ylabel(ylabel=f"Share of the average per {BLOCK_ITERATIONS} iterations (%)", fontsize=13)
    axis.set_title(label=f"What the averaged weights of iteration {last_iteration:,} are made of", fontsize=14)
    axis.grid(alpha=0.3)
    axis.legend(loc="upper left", fontsize=11)
    print(f"Iteration {last_iteration}: last update decay {last_update_decay:.6f}, window {last_update_window}, "
          f"its share {window_share:.2f} %, last {configured_window} share {configured_window_share:.2f} %, "
          f"starting weights {100.0 * trainer_start_share:.3g} %; constant decay: last {configured_window} share "
          f"{constant_configured_share:.2f} %, starting weights {100.0 * constant_start_share:.2f} %, newest block "
          f"{constant_block_shares[-1]:.2f} % against {trainer_block_shares[-1]:.2f} % with the warm-up")


def draw_window_panel(axis: plt.Axes, decay: float, last_iteration: int) -> None:
    """
    Draws the right panel: the window 1 / (1 - d) of the update of every iteration over a run long enough for the
    warm-up to end, the configured windows of the decay and of the smaller reference decay as grey lines, and the
    window at the last iteration of the left panel marked.

    Args:
        axis (plt.Axes): The axis of the panel.
        decay (float): The configured decay of the average.
        last_iteration (int): The iteration marked, the one of the left panel.
    """
    # Iteration at which the warm-up reaches the configured decay, (1 + n) / (10 + n) = decay
    configured_window = 1.0 / (1.0 - decay)
    warm_up_end = round(9.0 * configured_window)
    drawn_iterations = round(1.4 * warm_up_end)
    iterations, windows = compute_window_curve(decay=decay, last_iteration=drawn_iterations)
    axis.plot(iterations, windows, color=TRAINER_COLOUR, linewidth=2.5, label=f"decay {decay}")

    # Configured windows of the decay and of the reference decay, with the iteration the warm-up reaches them
    for reference_decay in [decay, REFERENCE_DECAY]:
        reference_window = 1.0 / (1.0 - reference_decay)
        reference_end = round(9.0 * reference_window)
        axis.axhline(y=reference_window, color=REFERENCE_COLOUR, linewidth=1.2, linestyle="--", zorder=0)
        axis.text(x=drawn_iterations, y=reference_window + 0.015 * configured_window,
                  s=f"1 / (1 − {reference_decay}) = {reference_window:,.0f}, reached at iteration {reference_end:,}",
                  ha="right", va="bottom", fontsize=11.5, color=TEXT_COLOUR)

    # Window of the last update of the left panel, labelled in the empty area above the warm-up line
    last_window = 1.0 / (1.0 - compute_update_decay(decay=decay, update_index=last_iteration - 1))
    axis.plot([last_iteration], [last_window], marker="o", markersize=9, color=TRAINER_COLOUR,
              markeredgecolor="white", markeredgewidth=2, zorder=3)
    axis.annotate(text=f"iteration {last_iteration:,}: window {last_window:,.0f}\n(left panel)",
                  xy=(last_iteration, last_window), xytext=(0.03 * drawn_iterations, 0.62 * configured_window),
                  ha="left", va="center", fontsize=12, color=TEXT_COLOUR,
                  arrowprops={"arrowstyle": "-", "color": TEXT_COLOUR, "linewidth": 1})

    # Axis labels, title and legend
    axis.set_xlim(left=0, right=drawn_iterations)
    axis.set_ylim(bottom=0, top=1.18 * configured_window)
    axis.xaxis.set_major_formatter(FuncFormatter(format_thousands))
    axis.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    axis.set_xlabel(xlabel="Iteration", fontsize=13)
    axis.set_ylabel(ylabel="Iterations averaged, 1 / (1 − decay of the update)", fontsize=13)
    axis.set_title(label="Iterations the average looks back over during a run", fontsize=14)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right", fontsize=11)


def plot_moving_average_window(output_path: Path, decay: float, last_iteration: int) -> None:
    """
    Draws both panels side by side and saves the image read by the README.

    Args:
        output_path (Path): The image to write.
        decay (float): The configured decay of the average.
        last_iteration (int): The iteration whose averaged weights the left panel decomposes.
    """
    figure, (composition_axis, window_axis) = plt.subplots(nrows=1, ncols=2, figsize=(16.0, 6.0), dpi=150)
    draw_composition_panel(axis=composition_axis, decay=decay, last_iteration=last_iteration)
    draw_window_panel(axis=window_axis, decay=decay, last_iteration=last_iteration)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    # Image path, decay and decomposed iteration from the command line, the detection launches by default
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--output_path", type=Path, required=True)
    argument_parser.add_argument("--decay", type=float, default=0.9998)
    argument_parser.add_argument("--last_iteration", type=int, default=15000)
    arguments = argument_parser.parse_args()
    plot_moving_average_window(output_path=arguments.output_path, decay=arguments.decay,
                               last_iteration=arguments.last_iteration)
