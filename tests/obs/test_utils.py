"""
Tests for the precipfm.obs.utils module.
"""
import numpy as np
import xarray as xr

from precipfm.obs.utils import track_stats


def test_track_stats(tmp_path):
    """
    Test tracking of observation statistics.
    """
    for ind in range(200):
        track_stats(tmp_path, "test", np.random.normal(size=(100, 100)))

    stats = xr.load_dataset(tmp_path / "stats.nc")
    mean = stats["test_sum"].data / stats["test_counts"].data
    assert np.isclose(mean, 0.0, atol=1e-2)

    squared_sum = stats["test_squared_sum"].data
    sum = stats["test_sum"].data
    cts = stats["test_counts"].data
    sigma = np.sqrt(squared_sum / cts - (sum / cts) ** 2)
    assert np.isclose(sigma, 1.0, atol=1e-2)

    assert stats["test_min"].data
    assert 0 < stats["test_max"].data
