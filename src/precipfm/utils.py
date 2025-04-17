"""
precipfm.utils
==============

General utils.
"""
from pathlib import Path
from datetime import datetime

import numpy as np


def get_date(path: Path) -> datetime:
    """
    Extract the date from any file stored from the precipfm package.

    Args:
        path: A path object pointing to a file with a filename of the format '..._%Y%m%d%H%M%S.nc'.

    Return:
        The file's timestamp parsed into a datetime object.
    """
    path = Path(path)
    parts = path.stem.split("_")
    if len(parts[-1]) == 2:
        date = datetime.strptime(parts[-2] + parts[-1], "%Y%m%d%H")
    else:
        date = datetime.strptime(parts[-1], "%Y%m%d%H%M%S")
    return date
