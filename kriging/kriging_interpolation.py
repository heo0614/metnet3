import torch
import numpy as np
import xarray as xr
import time
from typing import Optional, List
from dataclasses import dataclass
from kriging.functions import TorchKriging


@dataclass
class KrigingConfig:
    variogram_model: str = 'exponential'
    sampling_ratio: float = 0.025
    nugget: float = 0.0
    sill: Optional[float] = None
    range_param: Optional[float] = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    exclude_vars: List[str] = None  # Variables to exclude from processing


def dataset_to_array_kriging(ds: xr.Dataset, config: Optional[KrigingConfig] = None) -> np.ndarray:
    """
    Process all variables in an xarray Dataset using Kriging interpolation.
    
    Args:
        ds: xarray Dataset containing the variables to interpolate.
            If a numpy array (or MaskedArray) is provided, it will be converted to a DataArray.
        config: Optional KrigingConfig object for customization.
        
    Returns:
        Concatenated numpy array of interpolated values.
    """
    # 만약 ds가 xarray.Dataset가 아니라면, DataArray로 변환
    if not hasattr(ds, "coords"):
        # ds가 numpy 배열인 경우. ndim이 4이면 (time, channel, lat, lon)
        if ds.ndim == 4:
            dims = ("time", "channel", "lat", "lon")
            coords = {
                "time": np.arange(ds.shape[0]),
                "channel": np.arange(ds.shape[1]),
                "lat": np.arange(ds.shape[2]),
                "lon": np.arange(ds.shape[3]),
            }
        elif ds.ndim == 2:
            dims = ("lat", "lon")
            coords = {
                "lat": np.arange(ds.shape[0]),
                "lon": np.arange(ds.shape[1]),
            }
        else:
            raise ValueError(f"Unsupported array ndim: {ds.ndim}")
        ds = xr.DataArray(np.ma.filled(ds, np.nan), dims=dims, coords=coords).to_dataset(name="data")
    
    if config is None:
        config = KrigingConfig()
        
    device = torch.device(config.device)
    arrays = []
    
    # Get list of variables to process (제외할 변수와 좌표 변수는 건너뜀)
    exclude_vars = config.exclude_vars or []
    coordinate_vars = list(ds.coords.keys())
    var_list = [var for var in ds.variables 
                if var not in exclude_vars 
                and var not in coordinate_vars
                and len(ds[var].dims) >= 2]  # 최소 2차원 이상의 변수만 처리
    
    print(f"Processing variables: {var_list}")
    
    for var in var_list:
        da = ds[var]
        
        # spatial dimensions가 없으면 건너뜀
        if 'latitude' not in da.dims or 'longitude' not in da.dims:
            print(f"Skipping {var}: missing spatial dimensions")
            continue
            
        # 0인 값을 NaN으로 변경
        da = da.where(da != 0, np.nan)
        
        # xarray Dataset은 보통 'latitude'와 'longitude'라는 좌표를 갖습니다.
        # 이를 tensor로 변환
        lat = torch.tensor(da.latitude.values, device=device)
        lon = torch.tensor(da.longitude.values, device=device)
        
        # meshgrid 생성 (xy indexing)
        lon_grid, lat_grid = torch.meshgrid(lon, lat, indexing='xy')
        
        # 시간 축이 존재하면 각 시간에 대해 보간 처리, 아니면 2D 처리
        if 'time' in da.dims:
            time_arrays = []
            for time_idx in range(len(da.time)):
                values = torch.tensor(da[time_idx].values, device=device)
                interpolated = _interpolate_single_field(
                    values, lon_grid, lat_grid, config, device
                )
                time_arrays.append(interpolated)
            data_3d = np.stack(time_arrays, axis=0)  # (time, H, W)
        else:
            values = torch.tensor(da.values, device=device)
            data_3d = _interpolate_single_field(
                values, lon_grid, lat_grid, config, device
            )[np.newaxis, ...]
        
        # 만약 'level' 차원이 없다면, 새 차원을 추가하여 (time, 1, H, W) 형태로 만듦
        if 'level' not in da.dims:
            data_4d = data_3d[:, np.newaxis, :, :]
        else:
            data_4d = data_3d
            
        arrays.append(data_4d)
    
    return np.concatenate(arrays, axis=1)


def _interpolate_single_field(values: torch.Tensor, 
                              lon_grid: torch.Tensor, 
                              lat_grid: torch.Tensor, 
                              config: KrigingConfig,
                              device: torch.device) -> np.ndarray:
    """Helper function to interpolate a single 2D field."""
    # Flatten arrays
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()
    values_flat = values.flatten()
    
    # Sample valid points
    valid_mask = ~torch.isnan(values_flat)
    valid_points = valid_mask.sum().item()
    num_samples = int(config.sampling_ratio * valid_points)
    
    indices = torch.where(valid_mask)[0]
    sample_indices = indices[torch.randperm(len(indices))[:num_samples]]
    
    # Perform Kriging using TorchKriging
    kriging = TorchKriging(
        lon_flat[sample_indices],
        lat_flat[sample_indices],
        values_flat[sample_indices],
        variogram_model=config.variogram_model,
        nugget=config.nugget,
        sill=config.sill,
        range_param=config.range_param,
        device=config.device
    )
    
    # Predict on the entire grid
    predicted_values, _ = kriging.predict(lon_flat, lat_flat)
    
    # Reshape to original 2D shape and return as numpy array
    return predicted_values.reshape(values.shape).cpu().numpy()


# Example usage:
if __name__ == "__main__":
    # Load dataset (xarray.Dataset)
    ds = xr.open_dataset('/projects/aiid/KIPOT_SKT/Weather/sparse_data_input/156x156_sparse_0.5_input_all_80_zero.nc')
    
    # Optional: customize kriging parameters
    config = KrigingConfig(
        variogram_model='exponential',
        sampling_ratio=0.025,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        exclude_vars=['land_sea_mask']
    )
    
    start_time = time.time()
    interpolated_data = dataset_to_array_kriging(ds, config)
    elapsed_time = time.time() - start_time
    print(f"전체 동작 시간: {elapsed_time:.2f}초")
