"""
Tests for the precipfm.datasets module for loading the PrithviWxC input data.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from PrithviWxC.dataloaders.merra2 import (
    Merra2Dataset,
    preproc
)
import torch


from precipfm.datasets import MERRAInputData


MERRA_DATA_PATH = os.environ.get("MERRA_DATA", None)
HAS_MERRA_DATA = MERRA_DATA_PATH is not None


@pytest.mark.skipif(not HAS_MERRA_DATA, reason="MERRA inptu data not available.")
def test_merra_input_data():
    """
    Test that available input files for MERRA data are parsed correctly.
    """
    input_times = [-3, 0]
    lead_times = [3, 6]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_times=input_times,
        lead_times=lead_times
    )

    times_in_1 = dataset.times[dataset.input_indices[:, 0]]
    times_in_2 = dataset.times[dataset.input_indices[:, 1]]
    t_d = (times_in_2 - times_in_1).astype("timedelta64[h]").astype("int64")
    assert np.all(np.isclose(t_d, 3))

    times_out_1 = dataset.times[dataset.output_indices[:, 0]]
    times_out_2 = dataset.times[dataset.output_indices[:, 1]]
    t_d = (times_in_2 - times_in_1).astype("timedelta64[h]").astype("int64")
    assert np.all(np.isclose(t_d, 3))

    times_out_1 = dataset.times[dataset.input_indices[:, 1]]
    times_out_2 = dataset.times[dataset.output_indices[:, 0]]
    t_d = (times_in_2 - times_in_1).astype("timedelta64[h]").astype("int64")
    assert np.all(np.isclose(t_d, 3))

    input_times = [-3]
    lead_times = [-3]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_times=input_times,
        lead_times=lead_times
    )

    input_files = dataset.input_files[dataset.input_indices[:, 0]]
    output_files = dataset.input_files[dataset.output_indices[:, 0]]
    assert np.all(input_files == output_files)


@pytest.mark.skipif(not HAS_MERRA_DATA, reason="MERRA inptu data not available.")
def test_load_dynamic_data():
    """
    Test that available input files for MERRA data are parsed correctly.
    """
    input_times = [-3, 0]
    lead_times = [3, 6]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_times=input_times,
        lead_times=lead_times
    )

    data = dataset.load_dynamic_data(dataset.input_files[0])
    assert data.shape == (160, 361, 576)


@pytest.mark.skipif(not HAS_MERRA_DATA, reason="MERRA inptu data not available.")
def test_load_static_data():
    """
    Test that available input files for MERRA data are parsed correctly.
    """
    input_times = [-3, 0]
    lead_times = [3, 6]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_times=input_times,
        lead_times=lead_times
    )

    data = dataset.load_static_data(np.datetime64("2020-01-01T12:00:00"))
    assert data.shape == (10, 361, 576)


@pytest.mark.skipif(not HAS_MERRA_DATA, reason="MERRA inptu data not available.")
def test_load_static_data():
    """
    Test that available input files for MERRA data are parsed correctly.
    """
    input_times = [-3, 0]
    lead_times = [3, 6]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_times=input_times,
        lead_times=lead_times
    )

    data = dataset.load_static_data(np.datetime64("2020-01-01T12:00:00"))
    assert data.shape == (10, 361, 576)

@pytest.mark.skipif(not HAS_MERRA_DATA, reason="MERRA inptu data not available.")
def test_load_sample():
    """
    Test that available input files for MERRA data are parsed correctly.
    """
    input_times = [-3, 0]
    lead_times = [3, 6]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_times=input_times,
        lead_times=lead_times
    )
    x, y = dataset[0]

    assert "x" in x
    assert x["x"].shape == (2, 160, 360, 576)
    assert "climate" in x
    assert x["climate"].shape == (160, 360, 576)
    assert "static" in x
    assert x["static"].shape == (10, 360, 576)
    assert y.shape == (160, 360, 576)


def load_data_prithvi(data_path: Path, ind: int):
    """
    Load input data using original Prithvi implementation.
    """
    surf_dir = data_path / "merra-2"
    vert_dir = data_path / "merra-2"
    surf_clim_dir = data_path / "climatology"
    vert_clim_dir = data_path / "climatology"
    surface_vars = [
        "EFLUX", "GWETROOT", "HFLUX", "LAI", "LWGAB", "LWGEM", "LWTUP", "PS", "QV2M",
        "SLP", "SWGNT", "SWTNT", "T2M", "TQI", "TQL", "TQV", "TS", "U10M", "V10M", "Z0M",
    ]
    static_surface_vars = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
    vertical_vars = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
    levels = [34.0, 39.0, 41.0, 43.0, 44.0, 45.0, 48.0, 51.0, 53.0, 56.0, 63.0, 68.0, 71.0, 72.0,]
    lead_times = [6]
    input_times = [-6]
    time_range = ("2020-01-01T00:00:00", "2020-01-01T23:59:59")
    positional_encoding = "fourier"
    dataset = Merra2Dataset(
        time_range=time_range,
        lead_times=lead_times,
        input_times=input_times,
        data_path_surface=surf_dir,
        data_path_vertical=vert_dir,
        climatology_path_surface=surf_clim_dir,
        climatology_path_vertical=vert_clim_dir,
        surface_vars=surface_vars,
        static_surface_vars=static_surface_vars,
        vertical_vars=vertical_vars,
        levels=levels,
        positional_encoding=positional_encoding,
    )
    data = dataset[ind]

    padding = {"level": [0, 0], "lat": [0, -1], "lon": [0, 0]}
    return preproc([data], padding)


@pytest.mark.skipif(not HAS_MERRA_DATA, reason="MERRA input data not available.")
def test_loaded_data():
    input_times = [-6, 0]
    lead_times = [6]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_times=input_times,
        lead_times=lead_times
    )
    x, y = dataset[0]

    inpt_ref = load_data_prithvi(Path(MERRA_DATA_PATH).parent / "prithvi", 0)

    assert torch.all(torch.isclose(x["x"], inpt_ref["x"][0]))
    assert torch.all(torch.isfinite(x["x"]))
    assert torch.all(torch.isclose(y, inpt_ref["y"][0]))
    assert torch.all(torch.isclose(x["static"], inpt_ref["static"][0]))
    assert torch.all(torch.isclose(x["input_time"], inpt_ref["input_time"][0]))
    assert torch.all(torch.isclose(x["lead_time"], inpt_ref["lead_time"][0]))
    assert torch.all(torch.isclose(x["climate"], inpt_ref["climate"][0]))


@pytest.mark.skipif(not HAS_MERRA_DATA, reason="MERRA input data not available.")
def test_get_forecast_input_static():
    input_times = [-6, 0]
    lead_times = [6]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_times=input_times,
        lead_times=lead_times
    )

    static_data = dataset.get_forecast_input_static(np.datetime64("2020-01-01T06:00:00"), 4)
    assert static_data.shape == (5, 10, 360, 576)


@pytest.mark.skipif(not HAS_MERRA_DATA, reason="MERRA input data not available.")
def test_get_forecast_input_climate():
    input_times = [-6, 0]
    lead_times = [6]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_times=input_times,
        lead_times=lead_times
    )
    climate_data = dataset.get_forecast_input_climate(np.datetime64("2020-01-01T06:00:00"), 2)
    assert climate_data.shape == (3, 160, 360, 576)


@pytest.mark.skipif(not HAS_MERRA_DATA, reason="MERRA input data not available.")
def test_get_forecast_input_dynamic():
    input_times = [-6, 0]
    lead_times = [6]
    dataset = MERRAInputData(
        MERRA_DATA_PATH,
        input_times=input_times,
        lead_times=lead_times
    )
    x, y = dataset[0]
    dynamic_data = dataset.get_forecast_input_dynamic(np.datetime64("2020-01-01T06:00:00"))

    assert torch.all(x["x"] == dynamic_data)
