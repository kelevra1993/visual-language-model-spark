"""
Draws the two figures of the "Number Formats" subsection of the README Deployment section, for the four formats the
repository uses (FP32, TF32, BF16, FP16): README/number_format_examples.png puts six typical values of the repository
(a class threshold, a pixel coordinate, two box areas, a box offset and a weight update) on the number line of each
format, and README/number_format_rounding.png draws the gap between neighbouring values of each format from 1e-6 to
1e7. FP32, BF16 and FP16 values are rounded by PyTorch; TF32 rounds the FP32 mantissa to 10 bits, ties to even. From
the repository root:

    uv run python -m documentation.presentation.figures.draw_number_formats --output_folder README
"""

import argparse
import math
import matplotlib
import numpy
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path

from vision_training.utilities.os_utilities import format_path_link

# Every format: its mantissa bits, its smallest normal exponent, its largest value and its colour
FORMATS = {"FP32": {"mantissa_bits": 23, "minimum_exponent": -126, "largest_value": 3.4028234663852886e38,
                    "colour": "#2d3748"},
           "TF32": {"mantissa_bits": 10, "minimum_exponent": -126, "largest_value": 3.4028234663852886e38,
                    "colour": "#718096"},
           "BF16": {"mantissa_bits": 7, "minimum_exponent": -126, "largest_value": 3.3895313892515355e38,
                    "colour": "#7c3aed"},
           "FP16": {"mantissa_bits": 10, "minimum_exponent": -14, "largest_value": 65504.0, "colour": "#2563eb"}}

# Every example: its title, its exact value and the half width of the window drawn around it
EXAMPLES = [("class threshold: 0.999", 0.999, 0.0095), ("pixel coordinate of a 1024-pixel input: 1,000.3", 1000.3, 6.0),
            ("area of a 256 x 256 box: 65,536", 65536.0, 600.0), ("area of a 300 x 300 box: 90,000", 90000.0, 500.0),
            ("box offset: 0.1", 0.1, 0.0095), ("weight update: 1e-6", 1e-6, 9.5e-8)]

# Above this number of representable values in a window, a lane is drawn as a band instead of ticks
MAXIMUM_TICKS = 40


def round_to_format(value: float, format_name: str) -> float:
    """
    Rounds one value of the repository to the nearest value a number format holds, so that the README figures show
    what a served or trained model really stores (inf above the largest value).

    Args:
        value (float): The exact value.
        format_name (str): One of the keys of FORMATS.

    Returns:
        float: The stored value.
    """
    # PyTorch rounds FP32, BF16 and FP16 to nearest, ties to even
    if format_name in ("FP32", "BF16", "FP16"):
        dtype = {"FP32": torch.float32, "BF16": torch.bfloat16, "FP16": torch.float16}[format_name]
        return torch.tensor(data=value, dtype=torch.float64).to(dtype=dtype).item()
    # TF32 keeps 10 of the 23 FP32 mantissa bits, rounded to nearest, ties to even
    bits = numpy.array([value], dtype=numpy.float32).view(numpy.uint32)[0].item()
    dropped_bits = 13
    remainder, kept = bits & ((1 << dropped_bits) - 1), bits >> dropped_bits
    halfway = 1 << (dropped_bits - 1)
    if remainder > halfway or (remainder == halfway and kept & 1):
        kept += 1
    return numpy.array([kept << dropped_bits], dtype=numpy.uint32).view(numpy.float32)[0].item()


def gap_at(value: float, format_name: str) -> float:
    """
    Computes the gap between a value and the next value a format holds, the step that sets how far a stored value can
    move from the exact one (half of it at most).

    Args:
        value (float): A positive value.
        format_name (str): One of the keys of FORMATS.

    Returns:
        float: The gap at that value, constant below the smallest normal value (subnormals).
    """
    specification = FORMATS[format_name]
    exponent = max(math.floor(math.log2(value)), specification["minimum_exponent"])
    return 2.0 ** (exponent - specification["mantissa_bits"])


def representable_values(lower: float, upper: float, format_name: str) -> list[float]:
    """
    Lists the values a format holds between two bounds, to draw them as ticks on the lane of that format.

    Args:
        lower (float): The left bound of the window, positive.
        upper (float): The right bound of the window.
        format_name (str): One of the keys of FORMATS.

    Returns:
        list[float]: The values held, empty past the largest value, cut after MAXIMUM_TICKS + 1 values.
    """
    values, current = [], math.floor(lower / gap_at(value=lower, format_name=format_name)) * gap_at(
        value=lower, format_name=format_name)
    # Walk from value to value, the gap doubling at every power of two
    while current <= upper and len(values) <= MAXIMUM_TICKS:
        if current >= lower and current <= FORMATS[format_name]["largest_value"]:
            values.append(current)
        current += gap_at(value=max(current, lower), format_name=format_name)
    return values


def format_number(value: float) -> str:
    """
    Writes a stored value the way the README table does, with up to 6 significant digits and grouped thousands.

    Args:
        value (float): The stored value.

    Returns:
        str: The value as text.
    """
    if math.isinf(value):
        return "inf"
    if abs(value) >= 1000:
        return f"{value:,.1f}".rstrip("0").rstrip(".")
    return short_exponent(text=f"{value:.6g}")


def short_exponent(text: str) -> str:
    """
    Drops the leading zero of a printed exponent ("1e-06" becomes "1e-6"), as the README table writes them.

    Args:
        text (str): A number printed by Python.

    Returns:
        str: The same number with a short exponent.
    """
    return text.replace("e-0", "e-").replace("e+0", "e+")


def draw_examples(output_path: Path) -> None:
    """
    Draws the six example values on the number line of every format, one panel per value and one lane per format,
    with the stored value as a dot, the values the format holds as ticks and the gap written beside it.

    Args:
        output_path (Path): The PNG file written.
    """
    figure, axes_grid = plt.subplots(nrows=3, ncols=2, figsize=(18, 8.4))
    for axes, (title, exact_value, half_width) in zip(axes_grid.flat, EXAMPLES):
        lower, upper = exact_value - half_width, exact_value + half_width
        format_names = list(FORMATS)
        # One lane per format, FP32 at the top
        for lane_index, format_name in enumerate(format_names):
            lane_y, colour = len(format_names) - 1 - lane_index, FORMATS[format_name]["colour"]
            axes.plot([lower, upper], [lane_y, lane_y], color="#e2e8f0", linewidth=1.5, zorder=1)
            ticks = representable_values(lower=lower, upper=upper, format_name=format_name)
            # Too many values to tell apart: a band, otherwise one tick per value held
            if len(ticks) > MAXIMUM_TICKS:
                axes.plot([lower, upper], [lane_y, lane_y], color=colour, alpha=0.4, linewidth=5, zorder=2)
            else:
                axes.scatter(ticks, [lane_y] * len(ticks), marker="|", s=180, color=colour, zorder=2)
            stored_value = round_to_format(value=exact_value, format_name=format_name)
            # The stored value as a dot, or an arrow at the right end when it overflowed to inf
            if math.isinf(stored_value):
                axes.annotate("", xy=(upper, lane_y), xytext=(upper - half_width * 0.18, lane_y),
                              arrowprops={"arrowstyle": "-|>", "color": colour, "linewidth": 2})
                gap_text = f"above {FORMATS[format_name]['largest_value']:,.0f}"
            else:
                axes.scatter([stored_value], [lane_y], s=70, color=colour, edgecolor="white", zorder=3)
                gap_text = "gap " + short_exponent(text=f"{gap_at(value=exact_value, format_name=format_name):.3g}")
            # Stored value and gap written right of the lane, placed in axes fractions so panels share one layout
            axes.text(x=1.02, y=lane_y, s=format_number(value=stored_value), va="center", fontsize=12,
                      fontweight="bold", color=colour, transform=axes.get_yaxis_transform())
            axes.text(x=1.33, y=lane_y, s=gap_text, va="center", fontsize=10.5, color="#718096",
                      transform=axes.get_yaxis_transform())
        # Exact value as a dashed line, lanes named on the left
        axes.axvline(x=exact_value, color="#b91c1c", linestyle="--", linewidth=1.3, zorder=0)
        axes.set_yticks(range(len(format_names)), labels=list(reversed(format_names)), fontsize=11)
        axes.set_xticks([lower, exact_value, upper], labels=[format_number(value=lower),
                        format_number(value=exact_value), format_number(value=upper)], fontsize=10)
        axes.get_xticklabels()[1].set_color("#b91c1c")
        axes.set_xlim(lower, upper)
        axes.set_ylim(-0.6, len(format_names) - 0.4)
        axes.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#1a202c")
        axes.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
    figure.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.05, hspace=0.6, wspace=0.75)
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)


def draw_rounding(output_path: Path) -> None:
    """
    Draws the gap between neighbouring values of every format from 1e-6 to 1e7 on log axes, with a cross where FP16
    overflows and dotted lines at a class threshold, a pixel coordinate and a 256 x 256 box area.

    Args:
        output_path (Path): The PNG file written.
    """
    figure, axes = plt.subplots(figsize=(12, 6.6))
    magnitudes = numpy.logspace(-6, 7, 4000)
    for format_name, specification in FORMATS.items():
        kept = magnitudes[magnitudes <= specification["largest_value"]]
        gaps = [gap_at(value=magnitude, format_name=format_name) for magnitude in kept]
        # TF32 dashed under FP16, the two having the same gaps below 65,504
        line_style, z_order = ("--", 3) if format_name == "TF32" else ("-", 2)
        axes.step(kept, gaps, where="post", color=specification["colour"], linewidth=2.6, linestyle=line_style, zorder=z_order,
                  label=format_name + (" (same gaps as FP16, range of FP32)" if format_name == "TF32" else ""))
    axes.scatter([65504], [gap_at(value=65504.0, format_name="FP16")], marker="X", s=160, color=FORMATS["FP16"]["colour"],
                 zorder=4)
    axes.text(x=8e4, y=0.5, s="FP16 overflows\nabove 65,504", fontsize=12, fontweight="bold",
              color=FORMATS["FP16"]["colour"])
    # Dotted lines at three values of the repository
    for magnitude, label in [(0.999, "class threshold\n0.999"), (1024, "pixel coordinate\n1024"),
                             (65536, "area of a\n256 x 256 box")]:
        axes.axvline(x=magnitude, color="#4a5568", linestyle=":", linewidth=1.3)
        axes.text(x=magnitude * 1.15, y=2e-9, s=label, fontsize=11, color="#718096")
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlim(1e-6, 1e7)
    axes.set_ylim(1e-13, 1e5)
    axes.set_xlabel("magnitude of the value", fontsize=13)
    axes.set_ylabel("gap to the next representable value", fontsize=13)
    axes.grid(color="#edf2f7")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=11)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    # Folder of the two PNG files, README/ for the README
    parser = argparse.ArgumentParser(description="Draw the number format figures of the README.")
    parser.add_argument("--output_folder", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    for name, drawer in [("number_format_examples.png", draw_examples), ("number_format_rounding.png", draw_rounding)]:
        drawer(output_path=arguments.output_folder / name)
        print(f"Wrote {format_path_link(path=arguments.output_folder / name)}")
