"""
precipfm.grids
==============

Defines grid objects for resampling data to common grids.
"""
from pyresample import AreaDefinition


MERRA = AreaDefinition(
    area_id="MERRA2 grid",
    description="Regular lat/lon grid.",
    proj_id="merra",
    projection={
        "proj": "longlat",
        "datum": "WGS84",
    },
    width=576,
    height=360,
    area_extent=[-180.3125, -90, 179.6978, 90]
)
