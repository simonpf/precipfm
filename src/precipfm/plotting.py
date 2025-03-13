"""
precipfm.plotting
=================

General plotting functionality.
"""
from typing import Any, Dict, Optional, List

from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import numpy as np
import torch
import xarray as xr


def plot_forecast_results(
        results: xr.Dataset,
        variable: str,
        ax_width: int = 5,
        anomalies: bool = False,
        pcolormesh_kwargs: Dict[str, Any] = {},
        suptitle: Optional[str] = None
) -> plt.Figure:
    """
    Plot forecast results.

    Args:
        results: The xarray.Dataset containing the results.
        variable: The name of the variable to plot.
        anomalies: Whether or not to plot anomalies instead of absolute predictions.
    """
    pred = results[variable].data
    ref = results[variable + "_REF"].data

    if anomalies:
        pred = pred - results[variable + "_CLI"].data
        ref = ref - results[variable + "_CLI"].data

    v_min = min(pred.min(), ref.min())
    v_max = max(pred.max(), ref.max())

    n_steps = results.time.size

    fig = plt.figure(figsize=(ax_width * n_steps, 6))
    gs = GridSpec(2, n_steps + 2, width_ratios = [0.2] + [1.0] * n_steps + [0.075])

    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.text(0, 0, s="Forecast", rotation=90, ha="center", va="center")
    ax.set_ylim(-2, 2)
    ax.set_axis_off()

    ax = fig.add_subplot(gs[1, 0])
    ax.set_axis_off()
    ax.text(0, 0, s="Truth", rotation=90, ha="center", va="center")
    ax.set_ylim(-2, 2)
    ax.set_axis_off()

    lats = results.latitude.data
    lons = results.longitude.data

    for step in range(n_steps):

        ax = fig.add_subplot(gs[0, step + 1])
        ax.pcolormesh(lons, lats, pred[step], **pcolormesh_kwargs)
        if step > 0:
            ax.set_yticks([])
        ax.set_xticks([])

        time = results.time[step].data.astype("datetime64[s]").item()
        ax.set_title(time.strftime("%Y/%m/%d %H:%M"))
        ax.set_ylabel("Latitude [degree N]")

        ax = fig.add_subplot(gs[1, step + 1])
        m = ax.pcolormesh(lons, lats, ref[step], **pcolormesh_kwargs)
        if step > 0:
            ax.set_yticks([])

        ax.set_ylabel("Latitude [degree N]")
        ax.set_xlabel("Longitude [degree W]")

    cax = fig.add_subplot(gs[:, -1])
    label = variable
    if anomalies:
        label = f"$\Delta$ {variable}"
    plt.colorbar(m, cax=cax, label=label)

    if suptitle is not None:
        fig.suptitle(suptitle)

    return fig


def plot_tensors_as_animation(
        tensor_list: List[torch.Tensor],
        interval: int = 100,
        cmap: str = "magma",
        norm: Normalize = Normalize()
):
    """
    Plots a list of PyTorch tensors as an animation and displays it in a Jupyter notebook.

    Args:
        tensor_list: List of the results to plot.
        interval: Time interval between frames in milliseconds.
        cmap: Name of the colormap to use.
        norm: A normalization object to use to normalize the values.

    Return:
        A HTML animation object to display the results in a Jupyter notebook.
    """
    images = [t.detach().cpu().squeeze().numpy() for t in tensor_list]

    # Ensure images are in correct format (H, W) or (H, W, C)
    fig, ax = plt.subplots()
    img = ax.imshow(images[0], cmap=cmap if images[0].ndim == 2 else None, norm=norm)
    ax.axis("off")

    def update(frame):
        img.set_array(images[frame])
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=len(images), interval=interval, blit=False)
    plt.close(fig)
    return HTML(ani.to_jshtml())
