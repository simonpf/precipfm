"""
precipfm.forecast
=================

Provides functionality to perform forecasts with the PrithviWxC foundation model.
"""
import numpy as np
import torch
from torch import nn
import xarray as xr


from precipfm.merra import (
    SURFACE_VARS,
    VERTICAL_VARS
)


class Forecaster:
    """
    The forecaster class provides functionality to perform forecasts with the Prithvi WxFM.
    """
    def __init__(
            self,
            model: nn.Module,
            input_loader
    ):
        """
        Args:
            model: The PyTorch model to use to perform the forecast.
            input_loader: A data loader for the forecast input data.
        """
        self.model = model
        self.input_loader = input_loader


    def run(
            self,
            t_0: np.datetime64,
            n_steps: int,
            device: str = "cuda",
            dtype: str = "float32"
    ):
        """
        Run forecast for given initialization time 't_0'.

        Args:
            t_0: A datetime64 object specifying the initialization time for the forecast.
            n_steps: The number of forecast steps to run.
            device: String defining the device to run the fore cast on.
            dtype: The dtype to use for the forecast.
        """
        static_input = self.input_loader.get_forecast_input_static(t_0, n_steps)
        climate_input = self.input_loader.get_forecast_input_climate(t_0, n_steps)
        dynamic_input = self.input_loader.get_forecast_input_dynamic(t_0)
        lead_time = self.input_loader.lead_times[0]

        device = torch.device(device)
        dtype = getattr(torch, dtype)

        input_time = self.input_loader.input_times[1] - self.input_loader.input_times[0]

        x = {
            "x": dynamic_input[None].to(device=device, dtype=dtype),
            "lead_time": torch.tensor(lead_time)[None].to(device=device, dtype=dtype),
            "input_time": torch.tensor(input_time)[None].to(device=device, dtype=dtype)
        }
        model = self.model.to(dtype=dtype, device=device)

        results = []

        mr_tmp = model.mask_ratio_inputs

        with torch.no_grad():
            for step in range(n_steps):

                x["static"] = static_input[[step]].to(device=device, dtype=dtype)
                x["climate"] = climate_input[[step]].to(device=device, dtype=dtype)

                if step > 0:
                    model.mask_ratio_inputs = 0.0

                pred = model(x)


                results.append(pred.cpu().float().numpy()[0])
                x_new = torch.stack([x["x"][:, -1], pred], 1)
                x["x"] = x_new

        model.mask_ratio_inputs = mr_tmp

        clim = climate_input.cpu().numpy()
        results = np.stack(results, 0)
        times = t_0 + np.arange(1, n_steps + 1) * np.timedelta64(lead_time, "h")

        lons, lats = self.input_loader.get_lonlats()

        dataset = xr.Dataset({
            "time": (("time", times)),
            "latitude": (("latitude",), lats[:-1]),
            "longitude": (("longitude",), lons),
        })

        for var_ind, var in enumerate(SURFACE_VARS):
            dataset[var] = (("time", "latitude", "longitude"), results[:, var_ind])
        for var_ind, var in enumerate(VERTICAL_VARS):
            slc = slice(20 + var_ind * 14, 20 + (var_ind + 1) * 14)
            dataset[var] = (("time", "levels", "latitude", "longitude"), results[:, slc])

        for var_ind, var in enumerate(SURFACE_VARS):
            dataset[var + "_CLI"] = (("time", "latitude", "longitude"), clim[:, var_ind])
        for var_ind, var in enumerate(VERTICAL_VARS):
            slc = slice(20 + var_ind * 14, 20 + (var_ind + 1) * 14)
            dataset[var + "_CLI"] = (("time", "levels", "latitude", "longitude"), clim[:, slc])

        try:
            output = self.input_loader.get_forecast_output(t_0, n_steps)
            for var_ind, var in enumerate(SURFACE_VARS):
                dataset[var + "_REF"] = (("time", "latitude", "longitude"), output[:, var_ind])

            for var_ind, var in enumerate(VERTICAL_VARS):
                slc = slice(20 + var_ind * 14, 20 + (var_ind + 1) * 14)
                dataset[var + "_REF"] = (("time", "levels", "latitude", "longitude"), output[:, slc])
        except AttributeError as exc:
            raise exc

            pass

        return dataset
