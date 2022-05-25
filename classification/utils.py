import xarray as xr
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray, _get_time_indexes


def get_flat_output(data, filter = lambda data: data >= 1):
    flattened = filter(data).any(axis=1).astype(int)
    return numpy_to_xarray(flattened, data)


def get_reshaping(name="StandardScaler", horizon=None):
    def reshaping(x):
        if horizon is None:
            data = xr.DataArray(x.values.reshape((-1)), dims=["time"], coords={"time" : x[_get_time_indexes(x)[0]]})
        else:
            data = xr.DataArray(x.values.reshape((-1, horizon)), dims=["time", "horizon"], coords={"time" : x[_get_time_indexes(x)[0]],
                                                                                                   "horizon" : list(range(96))})
        return data

    return reshaping