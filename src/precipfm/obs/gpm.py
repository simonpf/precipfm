"""
precipfm.data.gpm
=================

This module provides functionality to extract satellite observations from the sensors of the
GPM constellation.
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
from pansat.products.satellite.gpm import (
    l1c_xcal2021v_f16_ssmis_v07a,
    l1c_xcal2021v_f16_ssmis_v07b,
    l1c_xcal2021v_f17_ssmis_v07a,
    l1c_xcal2021v_f17_ssmis_v07b,
    l1c_xcal2019v_noaa20_atms_v07a,
    l1c_xcal2019v_npp_atms_v07a,
    l1c_xcal2016v_noaa19_mhs_v07a,
    l1c_xcal2016v_noaa18_mhs_v07a,
    l1c_xcal2019v_metopc_mhs_v07a,
    l1c_xcal2016v_metopb_mhs_v07a,
    l1c_xcal2016c_gpm_gmi_v07a,
    l1c_xcal2016v_gcomw1_amsr2_v07a
)
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


_CHANNEL_REGEXP = re.compile("([\d\.]+)\s*(?:GHz)?(?:\+-)?\s*(?:\+\/-)?\s*([\d\.]*)\s*(?:GHz)?\s*(\w+)-Pol")


SSMIS_PRODUCTS = [
    l1c_xcal2021v_f16_ssmis_v07a,
    l1c_xcal2021v_f16_ssmis_v07b,
    l1c_xcal2021v_f17_ssmis_v07a,
    l1c_xcal2021v_f17_ssmis_v07b,
]

SENSORS = {
    "ssmis": [
        ("f16", 60e3, [l1c_xcal2021v_f16_ssmis_v07a, l1c_xcal2021v_f16_ssmis_v07b]),
        ("f17", 60e3, [l1c_xcal2021v_f17_ssmis_v07a, l1c_xcal2021v_f17_ssmis_v07b]),
    ],
    "atms": [
        ("noaa20", 60e3, [l1c_xcal2019v_noaa20_atms_v07a]),
        ("snpp", 60e3, [l1c_xcal2019v_npp_atms_v07a]),
    ],
    "mhs": [
        ("noaa19", 60e3, [l1c_xcal2016v_noaa19_mhs_v07a]),
        ("noaa18", 60e3, [l1c_xcal2016v_noaa18_mhs_v07a]),
        ("metop_b", 60e3, [l1c_xcal2016v_metopb_mhs_v07a]),
        ("metop_c", 60e3, [l1c_xcal2019v_metopc_mhs_v07a]),
    ],
    "gmi": [
        ("gpm", 20e3, [l1c_xcal2016c_gpm_gmi_v07a]),
    ],
    "amsr2": [
        ("gcomw1", 20e3, [l1c_xcal2016v_gcomw1_amsr2_v07a]),
    ],
}


def extract_observations(
        base_path: Path,
        gpm_file: FileRecord,
        target_grid: AreaDefinition,
        platform_name: str,
        sensor_name: str,
        radius_of_influence: float,
        n_tiles: Tuple[int, int] = (12, 18),
        time_step = np.timedelta64(3, "h"),
) -> xr.Dataset:
    """
    Extract observations from a GPM L1C file.

    Args:
        base_path: Path object pointing to the directory to which to write the extracted observations.
        gpm_file: A pansat FileRecord pointing to a GPM input file.
        platform_name: The name of the satellite platform the sensor is on.
        radius_of_incluence: Radius of influence to use for resampling.

    """
    l1c_obs = gpm_file.product.open(gpm_file)
    if "latitude" in l1c_obs:
        vars = [
            "latitude", "longitude", "tbs", "channels"
        ]
        l1c_obs = l1c_obs.rename(dict([(name, name + "_s1") for name in vars]))

    swath_ind = 1
    while f"latitude_s{swath_ind}" in l1c_obs:
        freqs = []
        offsets = []
        pols = []

        for match in _CHANNEL_REGEXP.findall(l1c_obs[f"tbs_s{swath_ind}"].attrs["LongName"]):
            freq, offs, pol = match
            freqs.append(float(freq))
            if offs == "":
                offsets.append(0.0)
            else:
                offsets.append(float(offs))
            pols.append(pol)

        swath_data = l1c_obs[[
            f"longitude_s{swath_ind}",
            f"latitude_s{swath_ind}",
            f"tbs_s{swath_ind}",
            f"channels_s{swath_ind}",
            "scan_time"
        ]].reset_coords("scan_time")

        fp_lons = swath_data[f"longitude_s{swath_ind}"].data
        fp_lats = swath_data[f"latitude_s{swath_ind}"].data
        sensor_lons = l1c_obs["spacecraft_longitude"].data
        sensor_lats = l1c_obs["spacecraft_latitude"].data
        sensor_alt = l1c_obs["spacecraft_altitude"].data * 1e3
        zenith, azimuth, viewing_angle = calculate_angles(
            fp_lons,
            fp_lats,
            sensor_lons,
            sensor_lats,
            sensor_alt
        )
        sensor_alt = np.broadcast_to(sensor_alt[..., None], zenith.shape) / 100e3

        swath_data = swath_data.rename({
            f"longitude_s{swath_ind}": "longitude",
            f"latitude_s{swath_ind}": "latitude"
        })
        swath_data["sensor_alt"] = (("scans", "pixels"), sensor_alt)
        swath_data["zenith"] = (("scans", "pixels"), zenith)
        swath_data["azimuth"] = (("scans", "pixels"), azimuth)
        swath_data["viewing_angle"] = (("scans", "pixels"), viewing_angle)

        swath_data_r = resample_data(
            swath_data,
            MERRA,
            radius_of_influence=radius_of_influence,
        ).transpose("latitude", "longitude", ...)
        sensor_alt = swath_data_r.sensor_alt.data
        zenith = swath_data_r.zenith.data
        azimuth = swath_data_r.azimuth.data
        viewing_angle = swath_data_r.viewing_angle.data

        start_time = swath_data.scan_time.min().data
        end_time = swath_data.scan_time.max().data
        start_time = round_time(start_time, time_step)
        end_time = round_time(end_time, time_step)
        times = np.arange(start_time, end_time + time_step, time_step)

        otime = swath_data_r.scan_time.data
        valid = np.isfinite(otime)

        for time in times:
            for chan_ind, (freq, offset, pol) in enumerate(zip(freqs, offsets, pols)):

                mask = (
                    (time <= swath_data_r.scan_time.data) *
                    (swath_data_r.scan_time.data <= (time + time_step))
                )
                if mask.sum() == 0:
                    continue

                obs = swath_data_r[f"tbs_s{swath_ind}"].data[..., chan_ind].copy()
                obs[obs < 0] = np.nan
                obs[obs > 400] = np.nan

                # Calculate relative time in seconds
                rel_time = (swath_data_r.scan_time.data - time).astype("timedelta64[s]").astype("float32")

                output = xr.Dataset({
                    "observations": (("y", "x"), obs),
                    "observation_relative_time": (("y", "x"), rel_time),
                    "observation_zenith_angle": (("y", "x"), zenith.copy()),
                    "observation_azimuth_angle": (("y", "x"), azimuth.copy()),
                })
                output["observations"].data[~mask] = np.nan
                output["observation_relative_time"].data[~mask] = np.datetime64("NAT")
                output["observation_zenith_angle"].data[~mask] = np.nan
                output["observation_azimuth_angle"].data[~mask] = np.nan

                output.attrs = {
                    "frequency": freq,
                    "wavelength": speed_of_light / (freq * 1e9) * 1e6,
                    "offset": offset,
                    "polarization": pol,
                }

                uint16_max = 2 ** 16 - 1
                encoding = {
                    "observations": {"dtype": "uint16", "scale_factor": 0.01, "_FillValue": uint16_max, "zlib": True},
                    "observation_relative_time": {"dtype": "uint16", "_FillValue": uint16_max, "zlib": True},
                    "observation_zenith_angle": {"dtype": "uint16", "scale_factor": 0.01, "_FillValue": uint16_max, "zlib": True},
                    "observation_azimuth_angle": {"dtype": "uint16", "scale_factor": 0.01, "_FillValue": uint16_max, "zlib": True},
                }

                n_rows, n_cols = output.observations.data.shape
                tile_dims = (n_rows // n_tiles[0], n_cols // n_tiles[1])

                obs_name = f"{platform_name}_{sensor_name}_{freq:.02f}_{offset:.02}_{pol.lower()}"

                valid = np.isfinite(output.observations.data)
                if valid.sum() == 0:
                    continue
                track_stats(base_path, obs_name, output.observations.data)

                output.attrs["obs_name"] = obs_name

                output = output.coarsen({"x": 32, "y": 30})
                output = output.construct({
                    "x": ("tiles_zonal", "lon_tile"),
                    "y": ("tiles_meridional", "lat_tile")
                })
                output = output.stack(tiles=("tiles_meridional", "tiles_zonal"))
                output = output.transpose("tiles", "lat_tile", "lon_tile")
                valid_tiles = np.isfinite(output.observations).mean(("lon_tile", "lat_tile")) > 0.25
                output = output[{"tiles": valid_tiles}].reset_index("tiles")

                obs_name = f"{platform_name}_{sensor_name}_{freq:.02f}_{offset:.02}_{pol.lower()}"
                output_file = obs_name + ".nc"
                output_path = base_path / get_output_path(time) / output_file
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if output_path.exists():
                    data = xr.load_dataset(output_path)
                    old_tiles = set(zip(data.tiles_meridional.data, data.tiles_zonal.data))
                    new_tiles = list(zip(output.tiles_meridional.data, output.tiles_zonal.data))
                    tile_mask = np.array([coords not in old_tiles for coords in new_tiles])
                    new_data = xr.concat((data, output[{"tiles": tile_mask}]), "tiles")
                    data.to_netcdf(output_path, encoding=encoding)
                else:
                    output.to_netcdf(output_path, encoding=encoding)


        swath_ind += 1


def extract_observations_day(
        sensor: str,
        year: int,
        month: int,
        day: int,
        output_path: Path
) -> None:
    """
    Extract GPM observations for a given day.

    Args:
        sensor: The name of the sensor.
        year: Integer specfiying the year.
        month: Integer specfiying the month.
        day: Integer specfiying the day.
        output_path: A path object pointing to the directory to which to write the extracted observations.
    """
    sensors = SENSORS[sensor]
    for platform_name, roi, pansat_products in sensors:
        for pansat_product in pansat_products:
            start = datetime(year, month, day)
            end = start + timedelta(days=1)
            recs = pansat_product.get(TimeRange(start, end))
            for rec in recs:
                try:
                    extract_observations(output_path, rec, MERRA, platform_name, sensor, roi)
                except:
                    LOGGER.exception(
                        "Encountered an error when processing input file %s.",
                        rec.local_path
                    )


@click.argument('sensor_name', type=str)
@click.argument('year', type=int)
@click.argument('month', type=int)
@click.argument('output_path', type=click.Path())
def process_sensor_data(sensor_name, year, month, output_path):
    """
    Process sensor data for a given sensor, year, and month, and save the result to the specified output path.

    SENSOR_NAME: Name of the sensor (string)
    YEAR: Year of the data to process (integer)
    MONTH: Month of the data to process (integer)
    OUTPUT_PATH: Path to save the processed data (string/path)
    """
    if month < 1 or month > 12:
        click.echo("Error: Month must be between 1 and 12.")
        return

    _, n_days = monthrange(year, month)
    for day in track(range(n_days)):
        extract_observations_day(sensor_name, year, month, day + 1, output_path)
