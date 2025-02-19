"""
precipfm.imerg
==============

Provides and interface to download IMERG data to fine tune the PrithviWxC foundation model.
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import binned_statistic_2d
from pansat import FileRecord, Geometry, TimeRange
from pansat.products.satellite.gpm import l3b_hhr_3imerg_ms_mrg_07b
import xarray as xr


LON_BINS = np.linspace(-180, 180, 577)
LAT_BINS = np.linspace(-90, 90, 181)


def download(year: int, month: int, day: int, output_path: Path) -> None:
    """
    Download IMERG data and resample to MERRA grid.

    Args:
        year: int specifying the year.
        month: int specifying the month
        day: int specifying the day.
        output_path: The path to which to write the extracted files.
    """
    output_path = Path(output_path)

    start_time = datetime(year, month, day, minute=1)
    end_time = start_time + timedelta(hours=23, minutes=58)
    time_range = TimeRange(start_time, end_time)
    recs = l3b_hhr_3imerg_ms_mrg_07b.get(time_range)

    precip_fields = []
    time = []

    for rec in recs:
        data = l3b_hhr_3imerg_ms_mrg_07b.open(rec).transpose("time", "latitude", "longitude")
        surface_precip = data.surface_precipitation.data
        lons = data.longitude.data
        lats = data.latitude.data
        lons, lats = np.meshgrid(lons, lats, indexing="xy")
        lons = lons[None]
        lats = lats[None]
        valid = 0.0 <= surface_precip
        surface_precip_r = binned_statistic_2d(
            lons[valid],
            lats[valid],
            surface_precip[valid],
            bins=(LON_BINS, LAT_BINS)
        )[0].T
        precip_fields.append(surface_precip_r)
        time.append(data.time.data[0])

    data = xr.Dataset({
        "latitude": 0.5 * (LAT_BINS[1:] + LAT_BINS[:-1]),
        "longitude": 0.5 * (LON_BINS[1:] + LON_BINS[:-1]),
        "time": np.stack(time),
        "surface_precip": (("time", "latitude", "longitude"), np.stack(precip_fields))
    })
    data = data.sortby("time")
    data = data.resample(time= "3h").mean()
    encoding = {"surface_precip": {"dtype": np.float32, "zlib": True}}
    for time_ind in range(data.time.size):
        data_t = data[{'time': time_ind}]
        time = data_t["time"].astype("datetime64[s]").item()
        fname = time.strftime("imerg_%Y%m%d%H%M.nc")
        data_t.to_netcdf(output_path / fname, encoding=encoding)
