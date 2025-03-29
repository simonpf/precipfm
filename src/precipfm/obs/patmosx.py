"""
precipfm.data.patmosx
=====================

This module provides functionality to extract observation data from the PATMOS-x dataset.
"""
from pathlib import Path
from typing import Tuple

from scipy.constants import speed_of_light
import numpy as np
from pansat.file_record import FileRecord
from pansat.time import to_datetime64
from pyresample import AreaDefinition
import xarray as xr

from .utils import (
    get_output_path,
    track_stats
)

OBS_VARS = [
    "temp_3_75um_nom",
    "temp_3_75um_nom_sounder",
    "temp_4_45um_nom_sounder",
    "temp_4_46um_nom",
    "temp_4_52um_nom",
    "temp_4_57um_nom_sounder",
    "temp_6_7um_nom",
    "temp_7_3um_nom",
    "temp_9_7um_nom",
    "temp_13_6um_nom",
    "temp_13_9um_nom",
    "temp_14_2um_nom",
    "temp_11_0um_nom",
    "temp_11_0um_nom",
    "temp_11_0um_nom_sounder",
    "temp_11_0um_nom",
    "temp_12_0um_nom",
    "temp_12_0um_nom_sounder",
    "temp_14_5um_nom_sounder",
    "temp_14_7um_nom_sounder",
    "temp_14_9um_nom_sounder",
    "temp_13_3um_nom",
    "temp_13_3um_nom",
    "refl_0_65um_nom",
    "refl_0_65um_nom",
    "refl_0_86um_nom",
    "refl_1_60um_nom",
    "refl_3_75um_nom",
]

def extract_observations(
        base_path: Path,
        patmosx_file: FileRecord,
        target_grid: AreaDefinition,
        n_tiles: Tuple[int, int] = (12, 18),
        time_step = np.timedelta64(3, "h"),
) -> xr.Dataset:
    """
    Extract observations from a PATMOX-x file.

    NOTE: This method assume that the target grid is a regular lat/lon grid.

    Args:
        base_path: Path object pointing to the directory to which to write the extracted observations.
        patmosx_file: A pansat FileRecord pointing to a PATMOS-x file.
        target_grid: The grid to which to write the input data.
        n_tiles: A tuple defining the number of meridional and zonal tiles.
        time_step: The time step by which to combine observations.
    """
    patmosx_file = patmosx_file.get()
    start_time = patmosx_file.temporal_coverage.start
    end_time = start_time + np.timedelta64(1, "D")
    time_steps = np.arange(start_time, end_time ,time_step)

    platform = patmosx_file.filename.split("_")[2].lower()

    lons, lats = target_grid.get_lonlats()
    lons = lons[0]
    lats = lats[:, 0]

    with xr.open_dataset(patmosx_file.local_path) as patmosx_data:
        obs_time = patmosx_data.scan_line_time[{"time": 0}].compute()
        obs_time = to_datetime64(start_time) + obs_time
        time_dtype = obs_time.dtype
        obs_time = obs_time.astype(np.int64).interp(latitude=lats, longitude=lons, method="nearest").astype(time_dtype)
        for obs_var in OBS_VARS:
            obs = patmosx_data[obs_var][{"time": 0}].compute()
            obs = obs.interp(latitude=lats, longitude=lons)

            for time in time_steps:
                step_start = time
                step_end = step_start + time_step
                mask = (step_start <= obs_time.data) * (obs_time.data < step_end)

                if mask.sum() == 0:
                    continue

                obs_t = obs.data.copy()
                obs_t[~mask] = np.nan

                # Calculate relative time in seconds
                rel_time = (obs_time.data - time).astype("timedelta64[s]").astype("float32")

                output = xr.Dataset({
                    "observations": (("y", "x"), obs_t),
                    "observation_relative_time": (("y", "x"), rel_time),
                })

                wl_1, wl_2 = obs_var.split("_")[1:3]
                wavelength = float(wl_1 + "." + wl_2[:-3])


                output.attrs = {
                    "frequency": speed_of_light / (wavelength / 1e6) / 1e9,
                    "wavelength": wavelength,
                    "offset": 0.0,
                    "polarization": "None",
                }

                uint16_max = 2 ** 16 - 1
                encoding = {
                    "observations": {"dtype": "uint16", "scale_factor": 0.01, "_FillValue": uint16_max, "zlib": True},
                    "observation_relative_time": {"dtype": "uint16", "_FillValue": uint16_max, "zlib": True},
                }

                n_rows, n_cols = output.observations.data.shape
                tile_dims = (n_rows // n_tiles[0], n_cols // n_tiles[1])

                obs_name = f"patmosx_{platform}_{obs_var}"

                valid = np.isfinite(output.observations.data)
                if valid.sum() == 0:
                    continue
                track_stats(base_path, obs_name, output.observations.data)

                output.attrs["obs_name"] = obs_name

                output = output.coarsen({"y": tile_dims[0], "x": tile_dims[1]})
                output = output.construct({
                    "x": ("tiles_zonal", "lon_tile"),
                    "y": ("tiles_meridional", "lat_tile")
                })
                output = output.stack(tiles=("tiles_meridional", "tiles_zonal"))
                output = output.transpose("tiles", "lat_tile", "lon_tile")
                valid_tiles = np.isfinite(output.observations).mean(("lon_tile", "lat_tile")) > 0.25
                output = output[{"tiles": valid_tiles}].reset_index("tiles")

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



def extract_observations_day(
        year: int,
        month: int,
        day: int,
        output_path: Path
):
    time = datetime(year, month, day, hour=12)
    recs = patmosx.find_files(time)
    LOGGER.info(
        f"Found {len(recs)} for {year:04}/{month:02}/{day:02}."
    )
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
@click.option("n_processes", help="The number of process to use for the data extraction.")
def extract_observations(
        year:int ,
        month: int,
        output_path: Path,
        n_processes: int = 1
):
    """
    Extract PATMOS-x  data for a given year, and month, and save the result to the specified output path.

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

    if n_processes > 1:
        LOGGER.info(f"[bold blue]Using {n_processes} processes for downloading data.[/bold blue]")
        tasks = [(year, month, d, output_path) for d in days]

        with ProcessPoolExecutor(max_workers=n_processes) as executor, Progress(console=console) as progress:
            task_id = progress.add_task("Extracting data:", total=len(tasks))
            future_to_task = {executor.submit(extract_observations_day, *task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as e:
                    logger.exception(f"Task {task} failed with error: {e}")
                finally:
                    progress.update(task_id, advance=1)
    else:
        with Progress(console=console) as progress:
            task_id = progress.add_task("Extracting data:", total=len(days))
            for d in days:
                try:
                    extract_observations_day(year, month, d, output_path)
                except Exception as e:
                    logger.exception(f"Error processing day {d}: {e}")
                finally:
                    progress.update(task_id, advance=1)
