"""
precipfm.geos
=============

Provides an interface to download GEOS reanalysis data.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path

import click
import numpy as np
from pansat.time import TimeRange, to_datetime64, to_datetime
from pansat.products.model.geos import (
    inst3_3d_asm_nv,
    inst3_2d_asm_nx,
    tavg1_2d_lnd_nx,
    tavg1_2d_flx_nx,
    tavg1_2d_rad_nx,
    tavg1_2d_flx_nx_fc,
)
import xarray as xr

from .merra import LEVELS, SURFACE_VARS, VERTICAL_VARS, NAN_VALS


DYNAMIC_PRODUCTS = [
    inst3_3d_asm_nv,
    inst3_2d_asm_nx,
    tavg1_2d_lnd_nx,
    tavg1_2d_flx_nx,
    tavg1_2d_rad_nx,
]


LOGGER = logging.getLogger(__name__)


def download_dynamic(year: int, month: int, day: int, output_path: Path) -> None:
    """
    Download dynamic GEOS input data for a date given by year, month, and day.

    Args:
        year: The year
        day: The day
        output_path: A path object pointing to the directory to which to download the data.
    """
    start_time = datetime(year, month, day)
    time_range = TimeRange(start_time, start_time + timedelta(hours=23, minutes=59))
    geos_recs = []
    for prod in DYNAMIC_PRODUCTS:
        prod_recs = prod.get(time_range)
        geos_recs.append(prod_recs)

    start_time = to_datetime64(datetime(year, month, day))
    end_time = start_time + np.timedelta64(1, "D")
    time_steps = np.arange(start_time, end_time, np.timedelta64(3, "h"))

    vars_req = VERTICAL_VARS + SURFACE_VARS

    all_data = []
    for recs in geos_recs:
        data_combined = []
        for rec in recs:
            with xr.open_dataset(rec.local_path) as data:
                vars = [
                    var for var in vars_req if var in data.variables
                ]
                data = data[vars + ["time"]]
                if "lev" in data:
                    data = data.loc[{"lev": np.array(LEVELS)}]
                data_combined.append(data.load())
        data = xr.concat(data_combined, "time").sortby("time")

        for var in data:
            if var in NAN_VALS:
                nan = NAN_VALS[var]
                data[var].data[:] = np.nan_to_num(data[var].data, nan=nan)


        if (data.time.data[0] - data.time.data[0].astype("datetime64[h]")) > 0:
            for var in data:
                data[var].data[1:] = 0.5 * (data[var].data[1:] + data[var].data[:-1])
            new_time = data.time.data - 0.5 * (data.time.data[1] -  data.time.data[0])
            data = data.assign_coords(time=new_time)

        times = list(data.time.data)
        time_steps = [step for step in time_steps if step in times]
        inds = [times.index(t_s) for t_s in time_steps]
        data_t = data[{"time": inds}]

        all_data.append(data_t)


    data = xr.merge(all_data, compat="override")
    data = data.rename(
        lat="latitude",
        lon="longitude"
    )

    output_path = Path(output_path) / f"{year:04}/{month:02}/{day:02}"
    output_path.mkdir(exist_ok=True, parents=True)

    encoding = {name: {"zlib": True} for name in data}

    for time_ind in range(data.time.size):
        data_t = data[{"time": time_ind}]
        date = to_datetime(data_t.time.data)
        output_file = date.strftime("geos_%Y%m%d%H%M%S.nc")
        data_t.to_netcdf(output_path / output_file, encoding=encoding)


@click.argument('year', type=int)
@click.argument('month', type=int)
@click.argument('day', type=int)
@click.argument('output_path', type=click.Path(writable=True))
def download_geos_forecast(year: int, month: int, day: int, output_path: Path) -> None:
    """
    Download GEOS precipitation forecasts results for a given day and year.

    Args:
        year: The year, if set to negative value will download forecasts from the previous day.
        day: The day
        output_path: A path object pointing to the directory to which to download the data.
    """
    if year < 1000:
        today = datetime.today() - timedelta(days=1)
        year = today.year
        month = today.month
        day = today.day

    start_time = datetime(year, month, day)
    end_time = start_time + timedelta(hours=23, minutes=59)
    start_time = to_datetime64(start_time)
    end_time = to_datetime64(end_time)
    init_times = np.arange(start_time, end_time, np.timedelta64(6, "h"))

    output_path = Path(output_path)
    output_folder = output_path / f"{year:04}" / f"{month:02}" / f"{day:02}"
    output_folder.mkdir(exist_ok=True, parents=True)

    for init_time in init_times:
        LOGGER.info(
            "Extracting forecasts for initialization time %s.",
            init_time
        )
        geos_recs = tavg1_2d_flx_nx_fc.get(
            TimeRange(init_time + np.timedelta64(3, "h"))
        )

        if len(geos_recs) == 0:
            LOGGER.info(
                "No forecasts found for initialization time %s.",
                init_time
            )
            continue

        geos_data = []
        for rec in geos_recs:
            with xr.open_dataset(rec.local_path) as data:
                data = data[["PRECTOT"]].compute()
                geos_data.append(data)

        geos_data = xr.concat(geos_data, dim="time")
        geos_data = geos_data.sortby("time")

        init_time = to_datetime(init_time)
        filename = init_time.strftime("geos_forecast_%Y%m%d_%H.nc")
        geos_data.to_netcdf(output_folder / filename)
