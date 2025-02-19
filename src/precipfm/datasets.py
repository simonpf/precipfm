"""
precipfm.datasets
=================

Provides dataset classes for loading input data for the PrithviWxC foundation model.
"""
from functools import cached_property, cache, partial
import os
import re
from pathlib import Path

from pansat.time import to_datetime64
import torch
from torch import nn
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from datetime import datetime
from typing import Tuple, Union, Optional, List

from precipfm.merra import (
    STATIC_SURFACE_VARS,
    SURFACE_VARS,
    VERTICAL_VARS
)


@cache
def load_static_data(root_dir: Path) -> xr.Dataset:
    """
    Load static input data from root data path.

    Args:
        root_dir: A path object containing the input data
    """
    static_file = root_dir / "static" / "merra2_static.nc"
    return xr.load_dataset(static_file)


def get_position_signal(lons: np.ndarray, lats:np.ndarray, kind: str) -> np.ndarray:
    """
    Calculate the position encoding.

    Args:
        lons: An array containing the longitude coordinates.
        lats: An array containing the latitude coordiantes.
        kind: A string defining the kind of the encoding. Currely supported are:
            - 'absolute': Returns the sine of the latitude coordinates and the cosine and sine
               of the longitude coordaintes stacked along the first dimensions
            - anything else: Simply returns the latitudes and longitudes in degree stacked along
              the first dimenion.
    """
    lons = lons.astype(np.float32)
    lats = lats.astype(np.float32)
    lons ,lats = np.meshgrid(lons, lats, indexing="xy")
    if kind == "absolute":
        lats_rad = np.deg2rad(lats_rad)
        lons_rad = np.deg2rad(lons_rad)
        static = np.stack([
            np.sin(lats_rad),
            np.cos(lons_rad),
            np.sin(lons_rad)
        ])
    return np.stack([lats, lons], axis=0).astype(np.float32)


def load_climatology(root_dir: Path, time: np.datetime64) -> np.ndarray:
    """
    Load climatology data.

    Args:
         root_dir: The root directory containing the foundation model data.
         time: A timestamp defining the time for which to load the input data.
    """
    date = time.astype("datetime64[s]").item()
    year = date.year
    doy = (date - datetime(year=year, month=1, day=1)).days + 1
    hod = date.hour

    sfc_file = root_dir / "climatology" / f"climate_surface_doy{doy:03}_hour{hod:02}.nc"
    data_sfc = []
    with xr.open_dataset(sfc_file) as sfc_data:
        for var in SURFACE_VARS:
            data_sfc.append(sfc_data[var].data.astype(np.float32))
    data_sfc = np.stack(data_sfc)

    data_vert = []
    vert_file = root_dir / "climatology" / f"climate_vertical_doy{doy:03}_hour{hod:02}.nc"
    with xr.open_dataset(vert_file) as vert_data:
        for var in VERTICAL_VARS:
            data_vert.append(np.flip(vert_data[var].data.astype(np.float32), 0))
    data_vert = np.stack(data_vert, 0)
    data_vert = data_vert.reshape(-1, *data_vert.shape[2:])

    data_combined = np.concatenate((data_sfc, data_vert), 0)
    return data_combined


class MERRAInputData(Dataset):
    """
    A PyTorch Dataset for loading 3-hourly MERRA2 data organized into year/month/day folders.
    """

    def __init__(
            self,
            root_dir: Union[Path, str],
            input_times: Optional[List[int]] = None,
            lead_times: Optional[List[int]] = None,
    ):

        """
        Args:
            root_dir (str): Root directory containing year/month/day folders.
        """
        if input_times is None:
            input_times = [-3]
        if lead_times is None:
            lead_times = [3]
        self.root_dir = Path(root_dir)
        self.times, self.input_files = self.find_files(self.root_dir)

        self._pos_sig = None
        self.input_times = input_times
        self.lead_times = lead_times
        self.input_indices, self.output_indices = self.calculate_valid_samples()

    def calculate_valid_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        A tuple of index arrays containing the indices of input- and output files for all training data
        samples satifying the requested input and lead time combination.

        Return: A tuple '(input_indices, output_indices)' with `input_indices` of shape
            '(n_samples, n_input_times)' containing the indices of all the input files for each data
            samples. Similarly, 'output_indices' is a numpy.ndarray of shape '(n_samples, n_lead_times)'
            containing the corresponding file indices to load for the output data.
        """
        input_indices = []
        output_indices = []
        for ind in range(self.times.size):
            sample_time = self.times[ind]
            input_times = [sample_time + np.timedelta64(t_i, "h") for t_i in self.input_times]
            lead_times = [sample_time + np.timedelta64(t_l, "h") for t_l in self.lead_times]
            valid = (
                all([t_i in self.times for t_i in input_times]) and
                all([t_l in self.times for t_l in lead_times])
            )
            if valid:
                input_indices.append([ind + t_i // 3 for t_i in self.input_times])
                output_indices.append([ind + t_l // 3 for t_l in self.lead_times])
        return np.array(input_indices), np.array(output_indices)

    def has_input(self, time: np.datetime64) -> bool:
        """
        Determine whether dynamic input for the given time stamp is available.
        """
        return time in self.times

    def get_lonlats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return longitude and latitude coordinates of MERRA2 data.
        """
        static = load_static_data(self.root_dir)
        lats = static.latitude.data
        lons = static.longitude.data
        return lons, lats

    def get_forecast_input_dynamic(
            self,
            initialization_time: np.datetime64
    ) -> torch.tensor:
        """
        Get dynamic forecast input data.

        Args:
            initialization_time: The initialization time for the forecast.

        Return:
            A torch tensor containing the model input data.
        """
        input_times = [initialization_time + np.timedelta64(t_i, "h") for t_i in self.input_times]
        for input_time in input_times:
            if not input_time in self.times:
                raise ValueError(
                    f"Input time {input_time} required for forecast but the data is not available."
                )

        input_file_inds = [self.times.searchsorted(input_time) for input_time in input_times]
        dynamic_in = [self.load_dynamic_data(self.input_files[ind]) for ind in input_file_inds]
        pad = partial(nn.functional.pad, pad=((0, 0, 0, -1)))
        dynamic_in = pad(torch.stack(dynamic_in, 0))
        return dynamic_in


    def get_forecast_output(
            self,
            initialization_time: np.datetime64,
            n_steps: int
    ):
        """
        Get forecast output for a given initialization time.
            Args:
                initialization_time: A np.datetime64 object specifying the forecast initialization time.
                n_steps: The number of forecasting steps.
        """
        lead_time = self.lead_times[0]
        input_times = [
            initialization_time + np.timedelta64(lead_time * (step + 1), "h") for step in range(n_steps)
        ]
        for input_time in input_times:
            if not input_time in self.times:
                raise ValueError(
                    f"Input time {input_time} required for forecast but the data is not available."
                )
        input_file_inds = [self.times.searchsorted(input_time) for input_time in input_times]
        dynamic_out = [self.load_dynamic_data(self.input_files[ind]) for ind in input_file_inds]
        pad = partial(nn.functional.pad, pad=((0, 0, 0, -1)))
        dynamic_out = pad(torch.stack(dynamic_out, 0))
        return dynamic_out


    def get_forecast_input_static(
            self,
            initialization_time: np.datetime64,
            forecast_steps: int
    ):
        """
        Get static forecast input.

        Returns static forecast input for all forecast steps.

        Args:
            initialization_time: The forecast initialization time.
            forecast_steps: The number of forecast steps.

        """
        time_steps = (
            initialization_time + (np.arange(forecast_steps) * self.lead_times[0]).astype("timedelta64[h]")
        )
        # Removes one row along lat dimension.
        pad = partial(nn.functional.pad, pad=((0, 0, 0, -1)))
        static_data = [pad(self.load_static_data(time)) for time in time_steps]
        static_data = torch.stack(static_data)
        return static_data

    def get_forecast_input_climate(
            self,
            initialization_time: np.datetime64,
            forecast_steps: int
    ):
        """
        Get climatology input for forecast.

        Args:
            initialization_time: The forecast initialization time.
            forecast_steps: The number of forecast steps.

        """
        time_steps = (
            initialization_time + (np.arange(1, forecast_steps + 1) * self.lead_times[0]).astype("timedelta64[h]")
        )
        # Removes one row along lat dimension.
        pad = partial(nn.functional.pad, pad=((0, 0, 0, -1)))
        climates = [pad(torch.tensor(load_climatology(self.root_dir, time))) for time in time_steps]
        return torch.stack(climates)

    def find_files(self, root_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gather all available MERRA2 files paths and extract available times.

        Args:
            root_dir: Path object pointing to the root of the data directory.

        Return:
            A tuple containing arrays of available inputs times and corresponding file
            paths.
        """
        times = []
        files = []
        pattern = re.compile(r"merra_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.nc")

        for path in sorted(list(root_dir.glob("**/merra2_*.nc"))):
            try:
                date = datetime.strptime(path.name, "merra2_%Y%m%d%H%M%S.nc")
                date64 = to_datetime64(date)

                files.append(str(path.relative_to(root_dir)))
                times.append(date64)
            except ValueError:
                continue

        times = np.array(times)
        files = np.array(files)
        return times, files

    def load_dynamic_data(self, path: Path) -> torch.Tensor:
        """
        Load all dynamic data from a given input file and return the data.

        Args:
            path: A path object pointing to the file to load.

        Return:
            A torch.Tensor containing all dynamic data for the given input file in the shape
            [var + levels (channels), lat, lon].
        """
        all_data = []
        with xr.open_dataset(self.root_dir / path) as data:
            for var in SURFACE_VARS:
                all_data.append(data[var].data[None].astype(np.float32))
            for var in VERTICAL_VARS:
                all_data.append(data[var].astype(np.float32))
        all_data = torch.tensor(np.concatenate(all_data, axis=0))
        return all_data

    def load_static_data(self, time: np.datetime64) -> torch.Tensor:
        """
        Load all dynamic data from a given input file and return the data.

        Args:
            path: A path object pointing to the file to load.

        Return:
            A torch.Tensor containing all dynamic data for the given
        """
        rel_time = time - time.astype("datetime64[Y]").astype(time.dtype)
        rel_time = np.datetime64("1980-01-01T00:00:00") + rel_time
        static_data = load_static_data(self.root_dir).interp(
            time=rel_time,
            method="nearest",
            kwargs={"fill_value": "extrapolate"}
        )
        lons = static_data.longitude.data
        lats = static_data.latitude.data

        if self._pos_sig is None:
            self._pos_sig = get_position_signal(lons, lats, kind="fourier")
        pos_sig = torch.tensor(self._pos_sig)

        n_time = 4
        n_pos = pos_sig.shape[0]
        n_lon = lons.size
        n_lat = lats.size
        n_static_vars = len(STATIC_SURFACE_VARS)

        data = torch.zeros((n_time + n_pos + n_static_vars, n_lat, n_lon))

        doy = time - time.astype("datetime64[Y]").astype(time.dtype)
        doy = doy.astype("timedelta64[D]").astype(int) + 1
        assert 0 <= doy <= 366

        hod = time - time.astype("datetime64[D]").astype(time.dtype)
        hod = hod.astype("timedelta64[h]").astype(int)
        assert 0 <= hod <= 24

        data[0:n_pos] = pos_sig
        data[n_pos + 0] = np.cos(2 * np.pi * doy / 366)
        data[n_pos + 1] = np.sin(2 * np.pi * doy / 366)
        data[n_pos + 2] = np.cos(2 * np.pi * hod / 24)
        data[n_pos + 3] = np.sin(2 * np.pi * hod / 24)
        #data[n_pos + 0] = np.cos(2 * np.pi * hod / 366)
        #data[n_pos + 1] = np.sin(2 * np.pi * hod / 366)
        #data[n_pos + 2] = np.cos(2 * np.pi * doy / 24)
        #data[n_pos + 3] = np.sin(2 * np.pi * doy / 24)

        for ind, var in enumerate(STATIC_SURFACE_VARS):
            data[n_pos + 4 + ind] = torch.tensor(static_data[var].data)

        return data

    def __len__(self):
        return len(self.input_indices)

    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single data point from the dataset.
        """
        input_files = [self.input_files[ind] for ind in self.input_indices[ind]]
        input_times = [self.times[ind] for ind in self.input_indices[ind]]
        output_files = [self.input_files[ind] for ind in self.output_indices[ind]]
        output_times = [self.times[ind] for ind in self.output_indices[ind]]

        dynamic_in = [self.load_dynamic_data(path) for path in input_files]
        static_in = self.load_static_data(input_times[-1])

        input_time = (input_times[1] - input_times[0]).astype("timedelta64[h]").astype(np.float32)
        lead_time = (output_times[0] - input_times[1]).astype("timedelta64[h]").astype(np.float32)

        dynamic_out = [self.load_dynamic_data(path) for path in output_files]
        climate = [load_climatology(self.root_dir, time) for time in output_times]

        # Remove one row along lat dimension.
        pad = partial(nn.functional.pad, pad=((0, 0, 0, -1)))

        x = {
            "x": pad(torch.stack(dynamic_in, 0)),
            "static": pad(static_in),
            "climate": pad(torch.tensor(climate[0])),
            "input_time": torch.tensor(input_time),
            "lead_time": torch.tensor(lead_time)
        }
        y = pad(torch.tensor(dynamic_out[0]))

        return x, y


class GEOSInputData(MERRAInputData):
    """
    A PyTorch Dataset for loading GEOS analysis data.
    """
    def find_files(self, root_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gather all available MERRA2 files paths and extract available times.

        Args:
            root_dir: Path object pointing to the root of the data directory.

        Return:
            A tuple containing arrays of available inputs times and corresponding file
            paths.
        """
        times = []
        files = []
        pattern = re.compile(r"geos_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.nc")

        for path in sorted(list(root_dir.glob("dynamic/**/geos_*.nc"))):
            try:
                date = datetime.strptime(path.name, "geos_%Y%m%d%H%M%S.nc")
                date64 = to_datetime64(date)

                files.append(str(path.relative_to(root_dir)))
                times.append(date64)
            except ValueError:
                continue

        times = np.array(times)
        files = np.array(files)
        return times, files


    def load_dynamic_data(self, path: Path) -> torch.Tensor:
        """
        Load all dynamic data from a given input file and return the data.

        Args:
            path: A path object pointing to the file to load.

        Return:
            A torch.Tensor containing all dynamic data for the given input file in the shape
            [var + levels (channels), lat, lon].
        """
        all_data = []
        with xr.open_dataset(self.root_dir / path) as data:
            for var in SURFACE_VARS:
                all_data.append(data[var].data[None].astype(np.float32))
            for var in VERTICAL_VARS:
                all_data.append(data[var].astype(np.float32))
        all_data = torch.tensor(np.concatenate(all_data, axis=0))
        return all_data

    def load_static_data(self, time: np.datetime64) -> torch.Tensor:
        """
        Load all dynamic data from a given input file and return the data.

        Args:
            path: A path object pointing to the file to load.

        Return:
            A torch.Tensor containing all dynamic data for the given
        """
        rel_time = time - time.astype("datetime64[Y]").astype(time.dtype)
        rel_time = np.datetime64("1980-01-01T00:00:00") + rel_time
        static_data = load_static_data(self.root_dir).interp(
            time=rel_time,
            method="nearest",
            kwargs={"fill_value": "extrapolate"}
        )
        lons = static_data.longitude.data
        lats = static_data.latitude.data

        if self._pos_sig is None:
            self._pos_sig = get_position_signal(lons, lats, kind="fourier")
        pos_sig = torch.tensor(self._pos_sig)

        n_time = 4
        n_pos = pos_sig.shape[0]
        n_lon = lons.size
        n_lat = lats.size
        n_static_vars = len(STATIC_SURFACE_VARS)

        data = torch.zeros((n_time + n_pos + n_static_vars, n_lat, n_lon))

        doy = time - time.astype("datetime64[Y]").astype(time.dtype)
        doy = doy.astype("timedelta64[D]").astype(int) + 1
        assert 0 <= doy <= 366

        hod = time - time.astype("datetime64[D]").astype(time.dtype)
        hod = hod.astype("timedelta64[h]").astype(int)
        assert 0 <= hod <= 24

        data[0:n_pos] = pos_sig
        data[n_pos + 0] = np.cos(2 * np.pi * doy / 366)
        data[n_pos + 1] = np.sin(2 * np.pi * doy / 366)
        data[n_pos + 2] = np.cos(2 * np.pi * hod / 24)
        data[n_pos + 3] = np.sin(2 * np.pi * hod / 24)

        for ind, var in enumerate(STATIC_SURFACE_VARS):
            data[n_pos + 4 + ind] = torch.tensor(static_data[var].data)

        return data

    def __len__(self):
        return len(self.input_indices)

    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single data point from the dataset.
        """

        input_files = [self.input_files[ind] for ind in self.input_indices[ind]]
        print("IF :: ", input_files)
        input_times = [self.times[ind] for ind in self.input_indices[ind]]
        output_files = [self.input_files[ind] for ind in self.output_indices[ind]]
        output_times = [self.times[ind] for ind in self.output_indices[ind]]

        dynamic_in = [self.load_dynamic_data(path) for path in input_files]
        static_in = self.load_static_data(input_times[-1])

        input_time = (input_times[1] - input_times[0]).astype("timedelta64[h]").astype(np.float32)
        lead_time = (output_times[0] - input_times[1]).astype("timedelta64[h]").astype(np.float32)

        dynamic_out = [self.load_dynamic_data(path) for path in output_files]
        climate = [load_climatology(self.root_dir, time) for time in output_times]

        # Remove one row along lat dimension.
        pad = partial(nn.functional.pad, pad=((0, 0, 0, -1)))

        x = {
            "x": pad(torch.stack(dynamic_in, 0)),
            "static": pad(static_in),
            "climate": pad(torch.tensor(climate[0])),
            "input_time": torch.tensor(input_time),
            "lead_time": torch.tensor(lead_time)
        }
        y = pad(torch.tensor(dynamic_out[0]))

        return x, y
