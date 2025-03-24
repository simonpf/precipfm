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
        n_tiles: Tuple[int, int] = (12, 18),
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
    cpcir_obs.attrs["offset"] = 0.0
    cpcir_obs.attrs["polarization"] = "None"

    cpcir_obs = cpcir_obs.coarsen({"longitude": 32, "latitude": 30})
    cpcir_obs = cpcir_obs.construct(
        {"longitude": ("tiles_zonal", "lon_tile"),
         "latitude": ("tiles_meridional", "lat_tile")}
    )
    cpcir_obs = cpcir_obs.stack(tiles=("tiles_meridional", "tiles_zonal"))
    cpcir_obs = cpcir_obs.transpose("tiles", "time", "lat_tile", "lon_tile")

    obs = cpcir_obs.observations.data
    obs[obs < 101] = np.nan
    obs[obs > 350] = np.nan

    for time_ind in range(cpcir_obs.time.size):

        cpcir_obs_t = cpcir_obs[{"time": time_ind}]
        time = cpcir_obs_t.time.data
        ref_time = round_time(time, time_step)
        rel_time = time - ref_time
        cpcir_obs_t.attrs["observation_relative_time"] = rel_time.astype("timedelta64[s]").astype("int64")

        valid = np.isfinite(cpcir_obs_t.observations.data)
        if valid.sum() == 0:
            continue
        track_stats(base_path, "cpcir", cpcir_obs_t.observations.data)
        cpcir_obs_t.attrs["obs_name"] = "cpcir"

        valid_tiles = np.isfinite(cpcir_obs_t.observations).mean(("lon_tile", "lat_tile")) > 0.5
        cpcir_obs_t = cpcir_obs_t[{"tiles": valid_tiles}].reset_index("tiles")

        rel_minutes = rel_time.astype("timedelta64[m]").astype("int64")
        obs_name = f"cpcir_{rel_minutes:03}"
        output_file = obs_name + ".nc"
        output_path = base_path / get_output_path(ref_time) / output_file

        encoding = {
            "observations": {
                "zlib": True,
                "dtype": "uint8",
                "add_offset": 100.0,
                "_FillValue": 255
            }
        }

        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)

        cpcir_obs_t.to_netcdf(output_path, encoding=encoding)


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
