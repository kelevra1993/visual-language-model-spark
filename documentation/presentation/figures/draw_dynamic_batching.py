"""
Draws the figure of the "Triton Or The Model In The Process" subsection of the README Deployment section
(README/dynamic_batching_requests.png): the same eight requests answered by the model in the process without Triton
(one call per image, one after another), by Triton with one instance and its dynamic batcher (the config.pbtxt that
export_model.py writes) and by Triton with two instances (instance_group count 2, set by hand). The schedules are
computed from the arrival times by schedule_in_process and schedule_with_triton, on a schematic time scale, not
measured. From the repository root:

    uv run python -m documentation.presentation.figures.draw_dynamic_batching --output_path README/dynamic_batching_requests.png
"""

import argparse
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List

from vision_training.utilities.os_utilities import format_path_link

# Schematic arrival times of the eight requests, in units of the time one image takes on the GPU
ARRIVAL_TIMES = [0.0, 0.2, 0.4, 1.0, 1.3, 1.45, 1.6, 2.6]

# Schematic costs: one image takes 1 unit, each more image of a batch adds 0.3, and the batcher waits at most 0.5 unit
# for other requests (max_queue_delay_microseconds, 1,000 in the exported config.pbtxt)
SINGLE_IMAGE_DURATION, EXTRA_IMAGE_DURATION, QUEUE_DELAY = 1.0, 0.3, 0.5

# Colours of the text, the arrivals, the latency arrows, the waiting brackets and the boxes of each panel
TEXT_COLOUR, ARRIVAL_COLOUR, LATENCY_COLOUR, WAIT_COLOUR = "#1f2937", "#2563eb", "#b91c1c", "#c2410c"
PANEL_COLOURS = ["#334155", "#7c3aed", "#0f766e"]


def schedule_in_process(arrival_times: List[float]) -> List[Dict]:
    """
    Computes when the model in the process answers each request: one call per image, a call starting only once the
    previous one has ended, as a Python process calling the model for every request does.

    Args:
        arrival_times (List[float]): The arrival time of every request, in order.

    Returns:
        List[Dict]: One run per request: its 'lane' (0), 'start', 'end' and the 'members' (request indices) it answers.
    """
    runs, previous_end = [], 0.0
    for request_index, arrival_time in enumerate(arrival_times):
        start = max(arrival_time, previous_end)
        runs.append({"lane": 0, "start": start, "end": start + SINGLE_IMAGE_DURATION, "members": [request_index]})
        previous_end = start + SINGLE_IMAGE_DURATION
    return runs


def schedule_with_triton(arrival_times: List[float], instance_count: int) -> List[Dict]:
    """
    Computes when Triton answers each request with its dynamic batcher: the oldest waiting request is held at most
    QUEUE_DELAY for others, then every request arrived by then runs as one batch on the first free instance; with two
    instances, two batches run at the same time.

    Args:
        arrival_times (List[float]): The arrival time of every request, in order.
        instance_count (int): The number of copies of the model on the GPU (instance_group count).

    Returns:
        List[Dict]: One run per batch: its 'lane' (instance index), 'start', 'end' and the 'members' it answers.
    """
    runs, waiting, free_times = [], list(range(len(arrival_times))), [0.0] * instance_count
    while waiting:
        # First free instance, starting once the delay of the oldest waiting request has passed
        instance_index = free_times.index(min(free_times))
        start = max(arrival_times[waiting[0]] + QUEUE_DELAY, free_times[instance_index])
        members = [request_index for request_index in waiting if arrival_times[request_index] <= start]
        end = start + SINGLE_IMAGE_DURATION + EXTRA_IMAGE_DURATION * (len(members) - 1)
        runs.append({"lane": instance_index, "start": start, "end": end, "members": members})
        free_times[instance_index] = end
        waiting = [request_index for request_index in waiting if request_index not in members]
    return runs


def describe_members(members: List[int]) -> str:
    """
    Writes the requests a box answers, numbered from 1 as on the arrival markers.

    Args:
        members (List[int]): The request indices, consecutive.

    Returns:
        str: 'request 8' for one request, 'requests 1-3' for several.
    """
    if len(members) == 1:
        return f"request {members[0] + 1}"
    return f"requests {members[0] + 1}-{members[-1] + 1}"


def draw_panel(axes: plt.Axes, top: float, title: str, lane_labels: List[str], runs: List[Dict], colour: str,
               show_wait: bool) -> None:
    """
    Draws one panel of the figure: its title, the numbered arrival of every request, one lane per model copy with a box
    per inference on the GPU labelled with the requests it answers, and the latency of request 8.

    Args:
        axes (plt.Axes): The axes of the figure, x in schematic time units and y in panel units.
        top (float): The y of the panel title.
        title (str): The title of the panel.
        lane_labels (List[str]): The label of every lane, from the top.
        runs (List[Dict]): The runs of schedule_in_process or schedule_with_triton.
        colour (str): The fill colour of the boxes.
        show_wait (bool): True to mark the wait of the first batch in the batcher queue.
    """
    axes.text(x=-0.1, y=top, s=title, ha="left", va="center", fontsize=17, fontweight="bold", color=TEXT_COLOUR)

    # Arrival of every request, its number above the marker
    arrival_y = top - 0.55
    for request_index, arrival_time in enumerate(ARRIVAL_TIMES):
        axes.plot(arrival_time, arrival_y, marker="v", markersize=13, color=ARRIVAL_COLOUR)
        axes.text(x=arrival_time, y=arrival_y + 0.28, s=str(request_index + 1), ha="center", va="center", fontsize=12,
                  fontweight="bold", color=ARRIVAL_COLOUR)

    # One lane per model copy, one box per inference with the requests it answers
    lane_tops = [arrival_y - 0.45 - 0.85 * lane_index for lane_index in range(len(lane_labels))]
    for lane_index, lane_label in enumerate(lane_labels):
        axes.text(x=-0.25, y=lane_tops[lane_index] - 0.3, s=lane_label, ha="right", va="center", fontsize=13,
                  color=TEXT_COLOUR)
    for run in runs:
        box_top = lane_tops[run["lane"]]
        axes.add_patch(Rectangle(xy=(run["start"], box_top - 0.6), width=run["end"] - run["start"] - 0.03, height=0.6,
                                 facecolor=colour, edgecolor="white", linewidth=1))
        axes.text(x=(run["start"] + run["end"]) / 2, y=box_top - 0.3, s=describe_members(members=run["members"]),
                  ha="center", va="center", fontsize=12, fontweight="bold", color="white")

    # Wait of the first batch in the batcher queue, from the arrival of request 1 to the start of its batch
    if show_wait:
        first_run = runs[0]
        wait_y = lane_tops[0] + 0.12
        axes.annotate(text="", xy=(first_run["start"], wait_y), xytext=(ARRIVAL_TIMES[0], wait_y),
                      arrowprops={"arrowstyle": "<->", "color": WAIT_COLOUR, "linewidth": 2})
        axes.text(x=first_run["start"] + 0.08, y=wait_y, s="waits at most 1 millisecond for other requests",
                  ha="left", va="center", fontsize=11, color=WAIT_COLOUR)

    # Latency of request 8, from its arrival to the end of the box answering it
    last_run = [run for run in runs if len(ARRIVAL_TIMES) - 1 in run["members"]][0]
    latency_y = lane_tops[-1] - 0.85
    axes.annotate(text="", xy=(last_run["end"], latency_y), xytext=(ARRIVAL_TIMES[-1], latency_y),
                  arrowprops={"arrowstyle": "<->", "color": LATENCY_COLOUR, "linewidth": 2})
    axes.text(x=last_run["end"] + 0.1, y=latency_y, s="latency of request 8", ha="left", va="center", fontsize=13,
              fontweight="bold", color=LATENCY_COLOUR)


def draw_dynamic_batching(output_path: Path) -> None:
    """
    Draws the three panels of the README figure, the model in the process without Triton, Triton with one instance and
    Triton with two instances, on one schematic time axis, and writes the PNG file.

    Args:
        output_path (Path): The PNG file to write.
    """
    figure, axes = plt.subplots(figsize=(20, 13))
    axes.set_xlim(-2.6, 9.4)
    axes.set_ylim(-0.6, 11.4)
    axes.axis("off")

    # The model in the process, without Triton: one call per image, one after another
    draw_panel(axes=axes, top=11.0,
               title="In the process, without Triton: one call per request, each call waiting for the previous one",
               lane_labels=["the model in\nthe process"], runs=schedule_in_process(arrival_times=ARRIVAL_TIMES),
               colour=PANEL_COLOURS[0], show_wait=False)

    # Triton with one instance, as export_model.py writes it: close requests run as one batch
    draw_panel(axes=axes, top=7.9,
               title="Triton, one instance (what export_model.py writes): requests arriving close together run as one "
                     "batch",
               lane_labels=["instance 1"], runs=schedule_with_triton(arrival_times=ARRIVAL_TIMES, instance_count=1),
               colour=PANEL_COLOURS[1], show_wait=True)

    # Triton with two instances, set by hand: two batches run at the same time on the GPU
    two_instance_runs = schedule_with_triton(arrival_times=ARRIVAL_TIMES, instance_count=2)
    draw_panel(axes=axes, top=4.8,
               title="Triton, two instances (count 2, set by hand): two copies of the model run two batches at once",
               lane_labels=["instance 1", "instance 2"], runs=two_instance_runs, colour=PANEL_COLOURS[2],
               show_wait=False)

    # Time when both copies run a batch, shaded behind their boxes
    overlap_start = min(run["start"] for run in two_instance_runs if run["lane"] == 1)
    overlap_end = min(run["end"] for run in two_instance_runs if run["lane"] == 0)
    axes.add_patch(Rectangle(xy=(overlap_start, 2.25), width=overlap_end - overlap_start, height=1.65,
                             facecolor="#fde68a", alpha=0.7, zorder=0))
    axes.text(x=(overlap_start + overlap_end) / 2, y=3.95, s="both copies busy", ha="center", va="center",
              fontsize=11, color=WAIT_COLOUR)

    # Time axis and legend
    axes.annotate(text="", xy=(9.2, -0.05), xytext=(-0.1, -0.05),
                  arrowprops={"arrowstyle": "->", "color": TEXT_COLOUR, "linewidth": 1.5})
    axes.text(x=4.5, y=-0.4,
              s="time (schematic: one image takes 1 unit on the GPU, each more image of a batch adds 0.3)",
              ha="center", va="center", fontsize=13, color="#6b7280")
    axes.plot(-2.4, 0.55, marker="v", markersize=13, color=ARRIVAL_COLOUR)
    axes.text(x=-2.2, y=0.55, s="a request arrives (its number)", ha="left", va="center", fontsize=12,
              color=TEXT_COLOUR)
    axes.add_patch(Rectangle(xy=(1.6, 0.4), width=0.5, height=0.3, facecolor=PANEL_COLOURS[0]))
    axes.text(x=2.2, y=0.55, s="one inference on the GPU, with the requests it answers", ha="left", va="center",
              fontsize=12, color=TEXT_COLOUR)

    figure.savefig(output_path, dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    # Path of the PNG file, README/dynamic_batching_requests.png for the README
    parser = argparse.ArgumentParser(description="Draw the dynamic batching figure of the README.")
    parser.add_argument("--output_path", type=Path, required=True)
    arguments = parser.parse_args()
    draw_dynamic_batching(output_path=arguments.output_path)
    print(f"Wrote {format_path_link(path=arguments.output_path)}")
