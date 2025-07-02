"""
precipfm.merra_precip
=====================

Provides an interface to extract surface precipitation estimates from MERRA2.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Union

import numpy as np
from pansat import FileRecord, Geometry, TimeRange
from pansat.products.reanalysis.merra import MERRA2
from pansat.time import to_datetime, to_datetime64
import xarray as xr

m2t1nxflx = MERRA2(
    collection="m2t1nxflx",
)


def download(year: int, month: int, day: int, output_path: Path) -> None:
    """
    Extract MERRA precip fields for a date given by year, month, and day.

    Args:
        year: The year
        day: The day
        output_path: A path object pointing to the directory to which to download the data.
    """
    output_path = Path(output_path)

    time_range = TimeRange(datetime(year, month, day, 12))
    merra_recs = m2t1nxflx.get(time_range)

    start_time = to_datetime64(datetime(year, month, day))
    end_time = start_time + np.timedelta64(1, "D")
    time_steps = np.arange(start_time, end_time, np.timedelta64(3, "h"))

    all_data = []
    for rec in merra_recs:
        with xr.open_dataset(rec.local_path) as data:
            data = data[["PRECTOT"]].load().rename(PRECTOT="surface_precip")
            data["surface_precip"].data *= 1e3
            all_data.append(data)

    data = xr.concat(all_data, "time").sortby("time")
    data = data.resample(time= "3h").mean()
    encoding = {"surface_precip": {"dtype": np.float32, "zlib": True}}

    for time_ind in range(data.time.size):
        data_t = data[{'time': time_ind}]
        time = to_datetime(data_t["time"].data)
        fname = time.strftime(f"prectot/{time.year:04}/{time.month:02}/prectot_%Y%m%d%H%M.nc")
        output_file = output_path / fname
        output_file.parent.mkdir(parents=True, exist_ok=True)
        data_t.to_netcdf(output_path / fname, encoding=encoding)
