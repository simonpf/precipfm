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


def plot_tiles(tnsr: torch.Tensor, global_y, global_x, local_y, local_x, channel: Optional[int] = None):
    """
    Plot tiled tensor.
    """
    tnsr = tnsr.detach().cpu()
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    gap = 3

    n_tiles_y = tnsr.shape[global_y]
    n_tiles_x = tnsr.shape[global_x]
    height = tnsr.shape[local_y]
    width = tnsr.shape[local_x]

    all_tiles = tnsr.squeeze()
    if channel is not None:
        all_tiles = all_tiles[..., channel, :, :]
    norm = Normalize(np.nanmin(all_tiles), np.nanmax(all_tiles))

    for tile_ind_y in range(n_tiles_y):
        for tile_ind_x in range(n_tiles_x):
            tile = tnsr.select(global_y, tile_ind_y).select(global_x - 1, tile_ind_x).squeeze()
            if channel is not None:
                tile = tile[channel]

            start = (width + gap) * tile_ind_x
            end = start + width
            x = np.arange(start, end)
            start = (height + gap) * tile_ind_y
            end = start + height
            y = np.arange(start, end)

            m = ax.pcolormesh(x, y, tile, norm=norm)

    plt.colorbar(m)


def plot_tiles_compressed(dataset: xr.Dataset):
    """
    Plot tiles xr.Dataset containing compressed tiles.
    """
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    gap = 3
    obs = dataset.observations.data
    _, height, width = obs.shape
    row_inds = dataset.tiles_meridional.data
    col_inds = dataset.tiles_zonal.data

    norm = Normalize(np.nanmin(obs), np.nanmax(obs))

    for obs_t, row_ind, col_ind in zip(obs, row_inds, col_inds):
        start = (width + gap) * col_ind
        end = start + width
        x = np.arange(start, end)
        start = (height + gap) * row_ind
        end = start + height
        y = np.arange(start, end)
        m = ax.pcolormesh(x, y, obs_t, norm=norm)

    plt.colorbar(m)


def set_style():
    """
    Set the IPWGML matplotlib style.
    """
    plt.style.use(Path(__file__).parent / "configs" / "precipfm.mplstyle")
