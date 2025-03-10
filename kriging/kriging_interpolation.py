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
    Process all variables in xarray Dataset using Kriging interpolation.
    
    Args:
        ds: xarray Dataset containing the variables to interpolate
        config: Optional KrigingConfig object for customization
        
    Returns:
        Concatenated numpy array of interpolated values
    """
    if config is None:
        config = KrigingConfig()
        
    device = torch.device(config.device)
    arrays = []
    
    # Get list of variables to process
    exclude_vars = config.exclude_vars or []
    coordinate_vars = list(ds.coords.keys())
    var_list = [var for var in ds.variables 
                if var not in exclude_vars 
                and var not in coordinate_vars
                and len(ds[var].dims) >= 2]  # Only process variables with spatial dimensions
    
    print(f"Processing variables: {var_list}")
    
    for var in var_list:
        # Get the data array and prepare it
        da = ds[var]
        
        # Skip variables without spatial dimensions
        if 'latitude' not in da.dims or 'longitude' not in da.dims:
            print(f"Skipping {var}: missing spatial dimensions")
            continue
            
        # Replace zeros with NaN
        da = da.where(da != 0, np.nan)
        
        # Convert coordinates to tensors
        lat = torch.tensor(da.latitude.values, device=device)
        lon = torch.tensor(da.longitude.values, device=device)
        
        # Create meshgrid
        lon_grid, lat_grid = torch.meshgrid(lon, lat, indexing='xy')
        
        # Handle different dimensional data
        if 'time' in da.dims:
            time_arrays = []
            for time_idx in range(len(da.time)):
                values = torch.tensor(da[time_idx].values, device=device)
                interpolated = _interpolate_single_field(
                    values, lon_grid, lat_grid, config, device
                )
                time_arrays.append(interpolated)
            data_3d = np.stack(time_arrays, axis=0)
        else:
            # Handle 2D data (single time step)
            values = torch.tensor(da.values, device=device)
            data_3d = _interpolate_single_field(
                values, lon_grid, lat_grid, config, device
            )[np.newaxis, ...]
        
        # Add level dimension if not present
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
    
    # Sample points for Kriging
    valid_mask = ~torch.isnan(values_flat)
    valid_points = valid_mask.sum().item()
    num_samples = int(config.sampling_ratio * valid_points)
    
    indices = torch.where(valid_mask)[0]
    sample_indices = indices[torch.randperm(len(indices))[:num_samples]]
    
    # Perform Kriging
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
    
    # Predict values
    predicted_values, _ = kriging.predict(lon_flat, lat_flat)
    
    # Reshape and return
    return predicted_values.reshape(values.shape).cpu().numpy()

#Example usage:
if __name__ == "__main__":
    # Load your dataset
    ds = xr.open_dataset('/projects/aiid/KIPOT_SKT/Weather/sparse_data_input/156x156_sparse_0.5_input_all_80_zero.nc')  # 사용될 dataset.
    
    # Optional: customize kriging parameters
    config = KrigingConfig(
        variogram_model='exponential',
        sampling_ratio=0.025,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        exclude_vars=['land_sea_mask']  # Optional: exclude specific variables
    )
    
    # 시간 측정 시작
    start_time = time.time()
    
    # Run interpolation on all suitable variables
    interpolated_data = dataset_to_array_kriging(ds, config)
    
    # 시간 측정 종료
    elapsed_time = time.time() - start_time
    print(f"전체 동작 시간: {elapsed_time:.2f}초")

