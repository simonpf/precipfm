"""
precipfm.data.cpcir
===================

This module provides functionality to extract geostationary 11 um satellite observations.
"""
from calendar import monthrange
import click
from datetime import datetime, timedelta
import logging
from pathlib import Path
import re
from typing import Tuple

from scipy.constants import speed_of_light
import numpy as np
from pansat import FileRecord
from pansat.time import to_datetime64, TimeRange
from pansat.utils import resample_data
from pansat.products.satellite.gpm import merged_ir
from pyresample.geometry import AreaDefinition
from rich.progress import track
from tqdm import tqdm
import xarray as xr

from ..grids import MERRA
from .utils import (
    calculate_angles,
    get_output_path,
    round_time,
    update_tile_list,
    track_stats
)


LOGGER = logging.getLogger(__name__)



def extract_observations(
        base_path: Path,
        cpcir_file: FileRecord,
        target_grid: AreaDefinition,
        n_tiles: Tuple[int, int] = (6, 9),
        time_step = np.timedelta64(3, "h"),
) -> xr.Dataset:
    """
    Extract observations from a GPM L1C file.

    Args:
        base_path: Path object pointing to the directory to which to write the extracted observations.
        gpm_file: A pansat FileRecord pointing to a GPM input file.
        radius_of_influence: Radius of influence to use for resampling.
    """
    cpcir_obs = cpcir_file.product.open(cpcir_file).rename(Tb="observations")
    lons, lats = target_grid.get_lonlats()
    lons = lons[0]
    lats = lats[:, 0]

    cpcir_obs = cpcir_obs.interp(latitude=lats, longitude=lons)

    wavelength = 11.0
    frequency = speed_of_light / (wavelength / 1e6) / 1e9
    cpcir_obs.attrs["wavelength"] = wavelength
    cpcir_obs.attrs["frequency"] = frequency
    cpcir_obs.attrs["polarization"] = "None"

    for time_ind in range(cpcir_obs.time.size):

        cpcir_obs_t = cpcir_obs[{"time": time_ind}]
        time = cpcir_obs_t.time.data
        ref_time = round_time(time, time_step)
        rel_time = time - ref_time
        cpcir_obs_t.attrs["relative_time"] = rel_time.astype("timedelta64[s]").astype("int64")

        height, width = cpcir_obs_t.observations.data.shape
        tile_dims = (height // n_tiles[0], width // n_tiles[1])

        track_stats(base_path, "cpcir", cpcir_obs_t.observations.data)
        cpcir_obs_t.attrs["name"] = "cpcir"

        for row_ind in range(n_tiles[0]):
            for col_ind in range(n_tiles[1]):
                row_start = row_ind * tile_dims[0]
                row_end = row_start + tile_dims[0]
                col_start = col_ind * tile_dims[1]
                col_end = col_start + tile_dims[1]

                tile = cpcir_obs_t[{"latitude": slice(row_start, row_end), "longitude": slice(col_start, col_end)}]
                valid_fraction = np.isfinite(tile.observations.data).mean()
                if valid_fraction < 0.3:
                    continue

                rel_minutes = rel_time.astype("timedelta64[m]").astype("int64")
                obs_name = f"{row_ind:02}_{col_ind:02}_cpcir_{rel_minutes:03}"
                output_file = obs_name + ".nc"
                output_path = base_path / get_output_path(ref_time) / output_file
                if not output_path.parent.exists():
                    output_path.parent.mkdir(parents=True)
                tile.to_netcdf(output_path)
                update_tile_list(output_path.parent, n_tiles, row_ind, col_ind, output_path.name)


def extract_observations_day(
        year: int,
        month: int,
        day: int,
        output_path: Path
):
    start = datetime(year, month, day)
    end = start + timedelta(days=1)
    recs = merged_ir.get(TimeRange(start, end))
    for rec in recs:
        try:
            extract_observations(output_path, rec, MERRA)
        except:
            LOGGER.exception(
                "Encountered an error when processing input file %s.",
                rec.local_path
            )


@click.argument('year', type=int)
@click.argument('month', type=int)
@click.argument('output_path', type=click.Path())
def process_data(year: int, month: int, output_path: str):
    """
    Extract geostationary observations for given year and month.

    YEAR: Year of the data to process (integer)
    MONTH: Month of the data to process (integer)
    OUTPUT_PATH: Path to save the processed data (string/path)
    """
    if month < 1 or month > 12:
        click.echo("Error: Month must be between 1 and 12.")
        return

    _, n_days = monthrange(year, month)
    for day in track(range(n_days)):
        extract_observations_day(year, month, day + 1, output_path)
