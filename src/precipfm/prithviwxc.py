"""
precipfm.prithviwxc
===================

Functions for loading the PrithviWxC model.
"""
from pathlib import Path
import toml
from typing import Any, Dict


import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from PrithviWxC.model import PrithviWxC
from PrithviWxC.dataloaders.merra2 import (
    input_scalers,
    output_scalers,
    static_input_scalers,
)
import torch
import torch.distributed
import torch.distributed.checkpoint


from precipfm.merra import (
    LEVELS,
    SURFACE_VARS,
    VERTICAL_VARS,
    STATIC_SURFACE_VARS
)


CONFIGS = {}


def load_configs() -> Dict[str, Dict[str, Any]]:
    """
    Load model configuration parameters.
    """
    config_files = sorted(list((Path(__file__).parent / "configs").glob("*.toml")))
    for config_file in config_files:
        with open(config_file) as inpt:
            CONFIGS[config_file.stem] = toml.load(inpt)

load_configs()


def load_model(
        checkpoint_path: Path | str,
        auxiliary_path: Path | str,
        configuration: str = "large",
        obs: bool = False
):
    """
    Load PrithviWxC model.

    Args:
        checkpoint_path: Path pointing to the checkpoint path.
        auxiliary_path: Path containing the auxiliary data, i.e., scalers etc.
        configuration: Name of the model configuration.

    Return:
        A pytorch module containing the loaded PrithviWxC model.
    """
    auxiliary_path = Path(auxiliary_path)


    kwargs = CONFIGS.get(configuration, None)
    if kwargs is None:
        raise ValueError(
            f"Model configuration '{configuration}' is not known. Currently known configurations are "
            f"{list(config_files.keys())}."
        )

    in_mu, in_sig = input_scalers(
        SURFACE_VARS,
        VERTICAL_VARS,
        LEVELS,
        auxiliary_path / "musigma_surface.nc",
        auxiliary_path / "musigma_vertical.nc",
    )

    output_sig = output_scalers(
        SURFACE_VARS,
        VERTICAL_VARS,
        LEVELS,
        auxiliary_path / "anomaly_variance_surface.nc",
        auxiliary_path / "anomaly_variance_vertical.nc",
    )

    static_mu, static_sig = static_input_scalers(
        auxiliary_path / "musigma_surface.nc",
        STATIC_SURFACE_VARS,
    )

    kwargs["input_scalers_mu"] = in_mu
    kwargs["input_scalers_sigma"] = in_sig
    kwargs["static_input_scalers_mu"] = static_mu
    kwargs["static_input_scalers_sigma"] = static_sig
    kwargs["output_scalers"] = output_sig ** 0.5
    kwargs["residual"] = "climate"
    kwargs["masking_mode"] = "local"
    kwargs["mask_ratio_inputs"] = 0.0
    kwargs["mask_ratio_targets"] = 0.0

    model = PrithviWxC(**kwargs)

    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        fabric = L.Fabric(strategy=FSDPStrategy())
        fabric.launch()
        model = fabric.setup(model)
        #model_state_dict = model.state_dict()
        state = {"model": model}
        fabric.load(checkpoint_path, state)
        #reader = torch.distributed.checkpoint.FileSystemReader(checkpoint_path)
        #torch.distributed.checkpoint.load_state_dict(
        #    state_dict=model_state_dict,
        #    storage_reader=reader
        #)
    else:
        state_dict = torch.load(checkpoint_path, weights_only=False)
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        model.load_state_dict(state_dict, strict=False)
    return model
