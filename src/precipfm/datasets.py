"""
precipfm.datasets
=================

Provides dataset classes for loading input data for the PrithviWxC foundation model.
"""
from datetime import datetime
from functools import cached_property, cache, partial
import logging
from math import ceil, trunc
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pansat.time import to_datetime64, to_datetime
from PrithviWxC.dataloaders.merra2 import (
    input_scalers,
    static_input_scalers,
)
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import yaml

from precipfm.merra import (
    LEVELS,
    STATIC_SURFACE_VARS,
    SURFACE_VARS,
    VERTICAL_VARS
)


LOGGER = logging.getLogger(__name__)


POLARIZATIONS = {
    "NONE": 0,
    "H": 1,
    "V": 2,
    "QH": 3,
    "QV": 4
}


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
    # No climatology for leap years :(.
    doy = min(doy, 365)
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
            climate: bool = True
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
        self.time_step = lead_times[0]
        self.climate = climate

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
        input_times = [initialization_time + np.timedelta64(t_i, "h") for t_i in self.times]
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

        for path in sorted(list(root_dir.glob("dynamic/**/merra2_*.nc"))):
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

    def load_dynamic_data(self, path: Path, slcs: Optional[Dict[str, slice]] = None) -> torch.Tensor:
        """
        Load all dynamic data from a given input file and return the data.

        Args:
            path: A path object pointing to the file to load.

        Return:
            A torch.Tensor containing all dynamic data for the given input file in the shape
            [var + levels (channels), lat, lon].
        """
        LOGGER.debug(
            "Loading dynamic input from file %s.",
            path
        )
        all_data = []
        if slcs is None:
            slcs = {}
        with xr.open_dataset(self.root_dir / path) as data:
            for var in SURFACE_VARS:
                all_data.append(data[var].__getitem__(slcs).data[None].astype(np.float32))
            for var in VERTICAL_VARS:
                all_data.append(data[var].__getitem__(slcs).astype(np.float32))
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
        LOGGER.debug(
            "Loading static input from for time %s.",
            time
        )
        rel_time = time - time.astype("datetime64[Y]").astype(time.dtype)
        rel_time = np.datetime64("1980-01-01T00:00:00") + rel_time
        static_data = load_static_data(self.root_dir)
        static_data = static_data.interp(
            time=rel_time.astype("datetime64[ns]"),
            method="nearest",
            kwargs={"fill_value": "extrapolate"}
        )
        lons = static_data.longitude.data
        lats = static_data.latitude.data

        if self._pos_sig is None:
            self._pos_sig = np.deg2rad(get_position_signal(lons, lats, kind="fourier"))
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

    def get_forecast_input(self, init_time: np.datetime64, n_steps: int) -> Dict[str, torch.Tensor]:
        """
        Get forecast input data to perform a continuous forecast.
        """
        input_times = [init_time + np.timedelta64(t_i * self.time_step, "h") for t_i in [-1, 0]]
        for input_time in input_times:
            if input_time not in self.times:
                raise ValueError(
                    "Required input data for t=%s not available.",
                    input_time
                )

        dynamic_in = []
        for input_time in input_times:
            ind = np.searchsorted(self.times, input_times[-1])
            dynamic_in.append(self.load_dynamic_data(self.input_files[ind]))

        static_time = self.times[-1]
        static_in = self.load_static_data(static_time)

        pad = partial(nn.functional.pad, pad=((0, 0, 0, -1)))
        dynamic_in = pad(torch.stack(dynamic_in, 0))[None].repeat(n_steps, 1, 1, 1, 1)
        static_in = pad(static_in)[None].repeat(n_steps, 1, 1, 1)
        input_time = self.input_times[0] * torch.ones(n_steps)
        lead_time = self.time_step * torch.arange(1, n_steps + 1).to(dtype=torch.float32)

        x = {
            "x": dynamic_in,
            "static": static_in,
            "lead_time": lead_time,
            "input_time": input_time,
        }

        if self.climate:
            output_times = [init_time + step * np.timedelta64(self.time_step, "h") for step in range(1, n_steps + 1)]
            climate = [torch.tensor(load_climatology(self.root_dir, time)) for time in output_times]
            climate = pad(torch.stack(climate))
            x["climate"] = climate

        return x

    def get_batched_forecast_input(
            self,
            init_time: np.datetime64,
            n_steps: int,
            batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Get forecast input data to perform a continuous forecast.
        """
        x = self.get_forecast_input(init_time, n_steps=n_steps)
        batch_start = 0
        n_samples = x["x"].shape[0]
        while batch_start < n_samples:
            batch_end = batch_start + batch_size
            yield {name: tnsr[batch_start:batch_end] for name, tnsr in x.items()}
            batch_start = batch_end


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




class PrecipDiagnosisDataset(MERRAInputData):
    """
    A PyTorch Dataset for loading MERRA2 input data and conincident  precip fields
    organized into year/month/day folders.
    """
    def __init__(
            self,
            root_dir: Union[Path, str],
            scene_size: int = 128,
            validation: bool = False,
            auxiliary_path: Optional[Union[str, Path]] = None
    ):
        """
        Args:
            root_dir (str): Root directory containing year/month/day folders.
            time_step: The forecast time step.
            n_steps: The number of forecast steps to perform.
        """
        self.root_dir = Path(root_dir)
        self.scene_size = scene_size
        self.validation = validation
        self._pos_sig = None

        self.input_times, self.input_files = self.find_merra_files(self.root_dir)
        self.output_times, self.output_files = self.find_precip_files(self.root_dir)

        self.input_indices, self.output_indices = self.calculate_valid_samples()
        self.init_rng()

        if auxiliary_path is not None:
            auxiliary_path = Path(auxiliary_path)
            in_mu, in_sig = input_scalers(
                SURFACE_VARS,
                VERTICAL_VARS,
                LEVELS,
                auxiliary_path / "musigma_surface.nc",
                auxiliary_path / "musigma_vertical.nc",
            )
            static_mu, static_sig = static_input_scalers(
                auxiliary_path / "musigma_surface.nc",
                STATIC_SURFACE_VARS,
            )
            self.mu_dynamic = in_mu
            self.sigma_dynamic = in_sig
            self.mu_static = static_mu
            self.sigma_static = static_sig
        else:
            self.mu_dynamic = None
            self.sigma_dynamic = None
            self.mu_static = None
            self.sigma_static = None

    def init_rng(self, w_id=0):
        """
        Initialize random number generator.

        Args:
            w_id: The worker ID which of the worker process..
        """
        if self.validation:
            seed = 42
        else:
            seed = int.from_bytes(os.urandom(4), "big") + w_id

        self.rng = np.random.default_rng(seed)
        self.n_workers = 1

    def worker_init_fn(self, w_id: int) -> None:
        """
        Initializes the worker state for parallel data loading.

        Args:
            w_id: The ID of the worker.
        """
        self.init_rng(w_id)
        winfo = torch.utils.data.get_worker_info()
        n_workers = winfo.num_workers
        self.n_workers = n_workers
        self.files = self.files[w_id::n_workers]
        self.pool = ThreadPoolExecutor(max_workers=1)


    def find_merra_files(self, root_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
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

        for path in sorted(list(root_dir.rglob("merra2_*.nc"))):
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


    def find_precip_files(self, root_dir) -> np.ndarray:
        """
        Find precip files for training.
        """
        times = []
        files = []
        pattern = re.compile(r"imerg_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.nc")

        for path in sorted(list(root_dir.glob("imerg/**/imerg_*.nc"))):
            try:
                date = datetime.strptime(path.name, "imerg_%Y%m%d%H%M%S.nc")
                date64 = to_datetime64(date)
                files.append(str(path.relative_to(root_dir)))
                times.append(date64)
            except ValueError:
                continue

        times = np.array(times)
        files = np.array(files)
        return times, files


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
        for ind in range(self.input_times.size):
            sample_time = self.input_times[ind]
            output_ind = np.searchsorted(self.output_times, sample_time)
            if output_ind < len(self.output_times) and sample_time == self.output_times[output_ind]:
                input_indices.append(ind)
                output_indices.append(output_ind)
        return np.array(input_indices), np.array(output_indices)

    def __len__(self):
        return len(self.input_indices)

    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single data point from the dataset.
        """
        try:

            scene_size = self.scene_size
            row_start = self.rng.integers(0, 361 - scene_size)
            col_start = self.rng.integers(0, 576 - scene_size)
            row_slice = slice(row_start, row_start + scene_size)
            col_slice = slice(row_start, row_start + scene_size)
            slcs = {
                "latitude": row_slice,
                "longitude": col_slice
            }

            input_ind = self.input_indices[ind]
            dynamic_in = self.load_dynamic_data(self.input_files[input_ind], slcs=slcs)
            static_in = self.load_static_data(self.input_times[input_ind])[..., row_slice, col_slice]

            if self.mu_dynamic is not None:
                dynamic_in = (dynamic_in - self.mu_dynamic[..., None, None]) / self.sigma_dynamic[..., None, None]
                static_in = torch.cat(
                    (static_in[:2], (static_in[2:] - self.mu_static[3:, None, None]) / self.sigma_static[3:, None, None]),
                    0
                )
                dynamic_in += 1e-2 * self.rng.normal(size=dynamic_in.shape)
                dynamic_in = dynamic_in.to(dtype=torch.float32)
                static_in += 1e-2 * self.rng.normal(size=static_in.shape)
                static_in = static_in.to(dtype=torch.float32)

            output_ind = self.output_indices[ind]
            with xr.open_dataset(self.root_dir / self.output_files[output_ind]) as precip_data:
                precip_data = precip_data.surface_precip[slcs].data


            return {"dynamic": dynamic_in, "static": static_in}, precip_data

        except Exception as exc:
            raise exc
            LOGGER.exception(
                "Encountered an error when load training sample %s. Falling back to another "
                " randomly-chosen sample.",
                ind
            )
            new_ind = np.random.randint(0, len(self))
            return self[new_ind]


class PrecipForecastDataset(MERRAInputData):
    """
    A PyTorch Dataset for loading 3-hourly MERRA2 data and corresponding precip fields
    organized into year/month/day folders.
    """
    def __init__(
            self,
            root_dir: Union[Path, str],
            time_step: int = 3,
            n_steps: int = 8
    ):
        """
        Args:
            root_dir (str): Root directory containing year/month/day folders.
            time_step: The forecast time step.
            n_steps: The number of forecast steps to perform.
        """
        self.input_time = -time_step
        self.time_step = time_step
        self.n_steps = n_steps

        self.root_dir = Path(root_dir)
        self.input_times, self.input_files = self.find_merra_files(self.root_dir)
        self.output_times, self.output_files = self.find_precip_files(self.root_dir)

        self._pos_sig = None
        self.time_step = time_step
        self.input_indices, self.output_indices = self.calculate_valid_samples()


    def find_merra_files(self, root_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
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

        for path in sorted(list(root_dir.rglob("merra2_*.nc"))):
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


    def find_precip_files(self, root_dir) -> np.ndarray:
        """
        Find precip files for training.
        """
        times = []
        files = []
        pattern = re.compile(r"imerg_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.nc")

        for path in sorted(list(root_dir.glob("imerg/**/imerg_*.nc"))):
            try:
                date = datetime.strptime(path.name, "imerg_%Y%m%d%H%M%S.nc")
                date64 = to_datetime64(date)
                files.append(str(path.relative_to(root_dir)))
                times.append(date64)
            except ValueError:
                continue

        times = np.array(times)
        files = np.array(files)
        return times, files


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
        for ind in range(self.input_times.size):
            sample_time = self.input_times[ind]
            input_times = [sample_time + np.timedelta64(t_i * self.time_step, "h") for t_i in [-1, 0]]
            output_times = [
                sample_time + np.timedelta64(t_i * self.time_step, "h") for t_i in np.arange(1, self.n_steps + 1)
            ]
            valid = (
                all([t_i in self.input_times for t_i in input_times]) and
                all([t_l in self.output_times for t_l in output_times])
            )
            output_ind = np.searchsorted(self.output_times, sample_time)
            if valid:
                input_indices.append([ind - self.time_step // 3, ind])
                output_indices.append([output_ind + (step + 1) * (self.time_step // 3) for step in np.arange(0, self.n_steps)])
        return np.array(input_indices), np.array(output_indices)

    def __len__(self):
        return len(self.input_indices)

    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single data point from the dataset.
        """
        try:
            input_files = [self.input_files[ind] for ind in self.input_indices[ind]]
            input_times = [self.input_times[ind] for ind in self.input_indices[ind]]
            dynamic_in = [self.load_dynamic_data(path) for path in input_files]

            static_times = [
                self.output_times[out_ind] - np.timedelta64(self.time_step, "h") for out_ind in self.output_indices[ind]
            ]
            static_in = [self.load_static_data(static_time) for static_time in static_times]

            climate = [torch.tensor(load_climatology(self.root_dir, time)) for time in self.output_times]

            input_time = self.input_time
            lead_time = self.time_step

            # Remove one row along lat dimension.
            pad = partial(nn.functional.pad, pad=((0, 0, 0, -1)))

            x = {
                "x": pad(torch.stack(dynamic_in, 0)),
                "static": pad(torch.stack(static_in, 0)),
                "climate": pad(torch.stack(climate, 0)),
                "input_time": torch.tensor(input_time).to(dtype=torch.float32),
                "lead_time": torch.tensor(lead_time).to(dtype=torch.float32)
            }

            output_files = [self.output_files[ind] for ind in self.output_indices[ind]]
            precip = []
            for path in output_files:
                with xr.open_dataset(self.root_dir / path) as data:
                    precip.append(torch.tensor(data.surface_precip.data.astype(np.float32)))

            return x, precip
        except Exception as exc:
            raise exc
            LOGGER.exception(
                "Encountered an error when load training sample %s. Falling back to another "
                " randomly-chosen sample.",
                ind
            )
            new_ind = np.random.randint(0, len(self))
            return self[new_ind]


class DirectPrecipForecastDataset(PrecipForecastDataset):
    """
    A PyTorch Dataset for loading precipitation forecast training data for direct forecasts without
    unrolling.
    """
    def __init__(
            self,
            root_dir: Union[Path, str],
            time_step: int = 3,
            max_steps: int = 24,
            climate: bool = True,
            augment: bool = True,
            sampling_rate: float = 1.0
    ):
        """
        Args:
            root_dir (str): Root directory containing year/month/day folders.
            time_step: The forecast time step.
            max_steps: The maximum number of timesteps to forecast precipitation.
            augment: Whether or not to augment the training data using random meridioanl flipping and
                 zonal rolls.
        """
        self.input_time = -time_step
        self.time_step = time_step
        self.max_steps = max_steps
        self.climate = climate
        self.augment = augment
        self.sampling_rate = sampling_rate

        self.root_dir = Path(root_dir)
        self.input_times, self.input_files = self.find_merra_files(self.root_dir)
        self.output_times, self.output_files = self.find_precip_files(self.root_dir)

        self._pos_sig = None
        self.time_step = time_step
        self.input_indices, self.output_indices = self.calculate_valid_samples()
        self.rng = np.random.default_rng(seed=42)


    def worker_init_fn(self, w_id: int) -> None:
        """
        Seeds the dataset loader's random number generator.
        """
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)


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
        for ind in range(self.input_times.size):
            sample_time = self.input_times[ind]
            input_times = [sample_time + np.timedelta64(t_i * self.time_step, "h") for t_i in [-1, 0]]
            output_times = [
                sample_time + np.timedelta64(t_i * self.time_step, "h") for t_i in np.arange(1, self.max_steps + 1)
            ]
            output_times = [t_o for t_o in output_times if t_o in self.output_times]
            valid = all([t_i in self.input_times for t_i in input_times])
            if valid and len(output_times) > 0:
                input_indices.append([ind - self.time_step // 3, ind])
                output_inds = []
                for output_time in output_times:
                    output_ind = np.searchsorted(self.output_times, output_time)
                    output_inds.append(output_ind)
                output_indices.append(output_inds + [-1] * (self.max_steps - len(output_inds)))
        return np.array(input_indices), np.array(output_indices)

    def __len__(self):
        return trunc(len(self.input_indices) * self.sampling_rate)

    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single data point from the dataset.
        """
        lower = trunc(ind / self.sampling_rate)
        upper = min(trunc((ind + 1) / self.sampling_rate), len(self.input_indices) - 1)
        if lower < upper:
            ind = self.rng.integers(lower, upper)
        else:
            ind = lower

        try:
            input_files = [self.input_files[ind] for ind in self.input_indices[ind]]
            input_times = [self.input_times[ind] for ind in self.input_indices[ind]]
            dynamic_in = [self.load_dynamic_data(path) for path in input_files]

            static_time = input_times[-1]
            static_in = self.load_static_data(static_time)

            input_time = self.input_time
            lead_time = self.time_step

            # Remove one row along lat dimension.
            pad = partial(nn.functional.pad, pad=((0, 0, 0, -1)))

            x = {
                "x": pad(torch.stack(dynamic_in, 0)),
                "static": pad(static_in),
                "input_time": torch.tensor(input_time).to(dtype=torch.float32),
                "lead_time": torch.tensor(lead_time).to(dtype=torch.float32)
            }

            inds = self.output_indices[ind]
            inds = inds[0 <= inds]
            output_ind = self.rng.choice(inds)
            output_file = self.output_files[output_ind]
            output_time = self.output_times[output_ind]
            lead_time = (output_time - max(input_times)).astype("timedelta64[h]").astype(np.float32)
            x["lead_time"] = torch.tensor(lead_time).to(dtype=torch.float32)

            if self.climate:
                climate = load_climatology(self.root_dir, output_time)
                x["climate"] = pad(torch.tensor(climate))

            with xr.load_dataset(self.root_dir / output_file) as data:
                LOGGER.debug("Loading precip data from %s.", output_file)
                precip = torch.tensor(data.surface_precip.data.astype(np.float32))

            coords = x["static"][:2]

            if self.augment:
                roll = self.rng.integers(0, 576 // 32) * 32
                for var in ["x", "static", "climate"]:
                    x[var] = x[var].roll(roll, dims=-1)


                flip = self.rng.random() > 0.5
                if flip:
                    for var in ["x", "static", "climate"]:
                        x[var] = x[var].flip(-2)

                x["static"][:2] = coords

            return x, precip

        except Exception as exc:
            raise exc
            LOGGER.exception(
                "Encountered an error when load training sample %s. Falling back to another "
                " randomly-chosen sample.",
                ind
            )
            new_ind = np.random.randint(0, len(self))
            return self[new_ind]


class ObservationLoader(Dataset):
    """
    PyTorch dataset for loading global satellite observations.
    """
    def __init__(
            self,
            observation_path: Path,
            observation_layers: int = 32,
            n_tiles: Tuple[int, int] = (12, 18),
            tile_size: Tuple[int, int] = (30, 32)
    ):
        """
        Args:
            observation_path: Path containing the observations.
        """
        self.observation_path = Path(observation_path)
        self.observation_layers = observation_layers
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        self.rng = np.random.default_rng(seed=42)
        self.time_step = 3
        self.freq_min = 1.0
        self.freq_max = 30e3
        self.file_regexp = None


    def worker_init_fn(self, w_id: int) -> None:
        """
        Seeds the dataset loader's random number generator.
        """
        #tracemalloc.start()
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)


    def find_files(self, path: Path) -> List[Path]:
        """
        Find all input files in YYY/mm/dd/HH folder.

        Args:
             path: The path in which to look for files.
        """

        files = np.random.permutation(sorted(list(path.glob("*.nc"))))
        if self.file_regexp is None:
            return files
        return [path for path in files if self.file_regexp.match(path.name)]

    def get_minmax(self, name: str) -> Tuple[float, float]:
        with xr.open_dataset(self.observation_path / "stats.nc") as stats:
            if f"{name}_min" not in stats:
                return np.nan, np.nan
            min_val = stats[f"{name}_min"].data
            max_val = stats[f"{name}_max"].data
        return min_val, max_val

    def load_observations(self, time: np.datetime64, offset: Optional[int] = None):
        """
        Load observations for a given time.
        """
        date = to_datetime(time)
        path = self.observation_path / date.strftime("%Y/%m/%d/obs_%Y%m%d%H%M%S.nc")

        observations = torch.nan * torch.zeros(self.n_tiles + (self.observation_layers, 1) + self.tile_size)
        meta_data = torch.nan * torch.zeros(self.n_tiles + (self.observation_layers, 8) + self.tile_size)

        if not path.exists():
            LOGGER.warning(
                "No observations for time %s.", time
            )
            return observations, meta_data

        layer_ind = np.zeros(self.n_tiles, dtype=np.int64)

        print("OBS :: ", path)
        data = xr.load_dataset(path)

        for row_ind in range(self.n_tiles[0]):
            for col_ind in range(self.n_tiles[1]):

                obs_name = f"observations_{row_ind:02}_{col_ind:02}"
                if obs_name not in data:
                    continue

                obs = torch.tensor(data[obs_name].data)
                inds = np.random.permutation(obs.shape[0])
                tiles = min(obs.shape[0], self.observation_layers)
                observations[row_ind, col_ind, :tiles, 0] = obs[inds[:tiles]]

                freq = np.log10(data[f"frequency_{row_ind:02}_{col_ind:02}"].data[inds[:tiles]])
                freq = -1.0 + 2.0 * (freq - np.log10(self.freq_max)) / (np.log10(self.freq_max) - np.log10(self.freq_min))
                offs = data[f"offset_{row_ind:02}_{col_ind:02}"].data[inds[:tiles]]
                pol = torch.nn.functional.one_hot(
                    torch.tensor(data[f"polarization_{row_ind:02}_{col_ind:02}"].data[inds[:tiles]]).to(dtype=torch.int64),
                    num_classes=5
                )

                time_offset = data[f"time_offset_{row_ind:02}_{col_ind:02}"].data[inds[:tiles]] / 180.0

                meta_data[row_ind, col_ind, :tiles, 0] = torch.tensor(freq)[..., None, None]
                meta_data[row_ind, col_ind, :tiles, 1] = torch.tensor(offs)[..., None, None]
                meta_data[row_ind, col_ind, :tiles, 2] = torch.tensor(time_offset)
                meta_data[row_ind, col_ind, :tiles, 3:] = pol[..., None, None]

        return observations, meta_data


class DirectPrecipForecastWithObsDataset(DirectPrecipForecastDataset):
    """
    A PyTorch Dataset for loading precipitation forecast training data for direct forecasts without
    unrolling.
    """
    def __init__(
            self,
            root_dir: Union[Path, str],
            time_step: int = 3,
            max_steps: int = 24,
            sentinel: float = -3.0,
            n_tiles: Tuple[int, int] = (12, 18),
            tile_size: Tuple[int, int] = (30, 32),
            climate: bool = False,
    ):
        """
        Args:
            root_dir (str): Root directory containing year/month/day folders.
            time_step: The forecast time step.
            max_steps: The maximum number of timesteps to forecast precipitation.
        """
        root_dir = Path(root_dir)
        super().__init__(root_dir=root_dir, time_step=time_step, max_steps=max_steps, climate=climate)
        self.rng = np.random.default_rng(seed=42)
        self.obs_loader = ObservationLoader(
            root_dir / "obs",
            n_tiles=n_tiles,
            tile_size=tile_size,
            observation_layers=32
        )

    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single data point from the dataset.
        """
        x, y = super().__getitem__(ind)
        input_times = [self.input_times[ind] for ind in self.input_indices[ind]]
        obs = []
        meta = []
        for time_ind, time in enumerate(input_times):
            obs_t, meta_t = self.obs_loader.load_observations(time, offset=len(input_times) - time_ind - 1)
            obs.append(obs_t)
            meta.append(meta_t)
        obs = torch.flip(torch.stack(obs, 0), (1, -2))
        obs_mask = torch.isnan(obs)
        obs = torch.nan_to_num(obs, nan=-1.5)
        meta = torch.stack(meta, 0)

        x["obs"] = obs
        x["obs_mask"] = obs_mask
        x["obs_meta"] = meta

        return x, y


class GEOSInputLoader(MERRAInputData):
    """
    A PyTorch Dataset for loading 3-hourly GEOS analysis data.
    """
    def __init__(
            self,
            root_dir: Union[Path, str],
            time_step: int = 3,
            n_steps: int = 8,
            climate: bool = True
    ):
        """
        Args:
            root_dir (str): Root directory containing year/month/day folders.
            time_step: The forecast time step.
            n_steps: The number of forecast steps to perform.
        """
        super().__init__(
            root_dir=root_dir,
            input_times=[-time_step],
            lead_times=[time_step]
        )
        self.input_time = -time_step
        self.time_step = time_step
        self.n_steps = n_steps
        self.climate = climate

        self.times, self.input_files = self.find_geos_files(self.root_dir)


    def find_geos_files(self, root_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
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

        for path in sorted(list(root_dir.rglob("geos_*.nc"))):
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





class GEOSObservationInputLoader(GEOSInputLoader):
    """
    A PyTorch Dataset for loading 3-hourly GEOS analysis data.
    """
    def __init__(
            self,
            root_dir: Union[Path, str],
            time_step: int = 3,
            n_steps: int = 8,
            climate: bool = True
    ):
        """
        Args:
            root_dir (str): Root directory containing year/month/day folders.
            time_step: The forecast time step.
            n_steps: The number of forecast steps to perform.
        """
        super().__init__(
            root_dir=root_dir,
            time_step=time_step,
            n_steps=n_steps,
            climate=climate
        )
        self.obs_loader = ObservationLoader(
            Path(root_dir).parent / "obs",
            n_tiles=(12, 18),
            tile_size=(30, 32),
            observation_layers=32
        )


    def get_forecast_input(self, init_time: np.datetime64) -> Dict[str, torch.Tensor]:
        """
        Get forecast input data to perform a continuous forecast.
        """
        x = super().get_forecast_input(init_time)
        input_times = [init_time + np.timedelta64(t_i * self.time_step, "h") for t_i in [-1, 0]]
        obs = []
        meta = []
        for time_ind, time in enumerate(input_times):
            obs_t, meta_t = self.obs_loader.load_observations(time, offset=len(input_times) - time_ind - 1)
            obs.append(obs_t)
            meta.append(meta_t)
        obs = torch.flip(torch.stack(obs, 0), (1, -2))
        obs_mask = torch.isnan(obs)
        obs = torch.nan_to_num(obs, nan=-1.5)
        meta = torch.stack(meta, 0)

        x["obs"] = obs[None].repeat_interleave(self.n_steps, 0)
        x["obs_mask"] = obs_mask[None].repeat_interleave(self.n_steps, 0)
        x["obs_meta"] = meta[None].repeat_interleave(self.n_steps, 0)

        return x
