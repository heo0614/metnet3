import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import Optional, List
from dataclasses import dataclass
from .functions import TorchKriging

@dataclass
class KrigingConfig:
    variogram_model: str = 'exponential'
    sampling_ratio: float = 0.025
    nugget: float = 0.0
    sill: Optional[float] = None
    range_param: Optional[float] = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    exclude_vars: List[str] = None  # Variables to exclude from processing

def dataset_to_array_kriging(ds: xr.Dataset, config: Optional[KrigingConfig] = None) -> xr.Dataset:
    """
    주어진 xarray Dataset에서 (time, level, lat, lon) 구조의 변수를 찾아 Kriging 보간을 수행하고,
    해당 변수를 Kriging된 값으로 치환한 새로운 Dataset을 반환합니다.
    - time, level 등 모든 차원을 유지한 채로 2D Kriging만 수행합니다.
    """
    if config is None:
        config = KrigingConfig()
    device = torch.device(config.device)

    # 결과 저장할 Dataset (원본 복사)
    ds_out = ds.copy(deep=True)
    
    # 처리에서 제외할 변수 목록, 좌표 목록
    exclude_vars = set(config.exclude_vars or [])
    coordinate_vars = set(ds_out.coords.keys())  # time, lat, lon, ...

    # 처리 대상 변수 추리기
    var_list = []
    for var in ds_out.data_vars:
        if var in exclude_vars:
            continue
        if var in coordinate_vars:
            continue
        da = ds_out[var]
        if 'latitude' not in da.dims or 'longitude' not in da.dims:
            continue
        var_list.append(var)
    
    # 각 변수별로 Kriging 진행
    for var in var_list:
        da = ds_out[var]
        # 0을 NaN으로 변환 (사용자 코드 로직)
        da = da.where(da != 0, np.nan)
        has_time = ('time' in da.dims)
        has_level = ('level' in da.dims)
        data_np = da.values  # 예: (time, level, lat, lon) 또는 (time, lat, lon)
        lat_vals = da['latitude'].values
        lon_vals = da['longitude'].values
        # meshgrid: imshow는 행-열 순서이므로 (lon, lat) 순서를 사용
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals, indexing='xy')
        
        if (not has_time) and (not has_level):
            data_np = _kriging_2d(data_np, lon_grid, lat_grid, config, device)
        elif has_time and (not has_level):
            nt = data_np.shape[0]
            for t in range(nt):
                slice_2d = data_np[t, :, :]
                data_np[t, :, :] = _kriging_2d(slice_2d, lon_grid, lat_grid, config, device)
        elif (not has_time) and has_level:
            nl = data_np.shape[0]
            for l in range(nl):
                slice_2d = data_np[l, :, :]
                data_np[l, :, :] = _kriging_2d(slice_2d, lon_grid, lat_grid, config, device)
        else:
            nt = data_np.shape[0]
            nl = data_np.shape[1]
            for t in range(nt):
                for l in range(nl):
                    slice_2d = data_np[t, l, :, :]
                    data_np[t, l, :, :] = _kriging_2d(slice_2d, lon_grid, lat_grid, config, device)
        
        ds_out[var].values = data_np
    return ds_out

def _kriging_2d(field_2d: np.ndarray,
                lon_grid: np.ndarray,
                lat_grid: np.ndarray,
                config: KrigingConfig,
                device: torch.device) -> np.ndarray:
    """
    단일 2D 필드(lat, lon)에 Kriging을 수행하여 보간된 2D numpy 배열을 반환.
    """
    field_t = torch.from_numpy(field_2d).to(device, dtype=torch.float)
    valid_mask = ~torch.isnan(field_t)
    if not valid_mask.any():
        return field_2d
    valid_indices = torch.nonzero(valid_mask.flatten()).squeeze(1)
    n_valid = len(valid_indices)
    n_sample = int(n_valid * config.sampling_ratio)
    if n_sample < 1:
        n_sample = 1
    perm = torch.randperm(n_valid, device=device)[:n_sample]
    sample_indices = valid_indices[perm]
    lon_flat = torch.from_numpy(lon_grid.ravel()).to(device, dtype=torch.float)
    lat_flat = torch.from_numpy(lat_grid.ravel()).to(device, dtype=torch.float)
    field_flat = field_t.flatten()
    kriging = TorchKriging(
        x=lon_flat[sample_indices],
        y=lat_flat[sample_indices],
        values=field_flat[sample_indices],
        variogram_model=config.variogram_model,
        nugget=config.nugget,
        sill=config.sill,
        range_param=config.range_param,
        device=config.device
    )
    pred_flat, _ = kriging.predict(lon_flat, lat_flat)
    pred_2d = pred_flat.view(*field_2d.shape).cpu().numpy()
    return pred_2d

