#%%

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
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

# --- 아래는 위치정보 및 그림 저장을 위한 코드 ---
def compute_masks(field: np.ndarray, hold_out_ratio=0.2, drop_out_ratio=0.25, seed: Optional[int] = None):
    """
    field: 2D numpy array (원본 이미지)
    hold_out: og_station 중 hold_out_ratio 만큼 랜덤 선택 (bool mask)
    drop_out: hold_out_pos에서 drop_out_ratio 만큼 랜덤 선택 (bool mask)
    """
    if seed is not None:
        np.random.seed(seed)
    # og_station: NaN이 아닌 위치
    og_station = ~np.isnan(field)
    # hold_out: og_station 중에서 20% 선택
    rand_vals = np.random.rand(*field.shape)
    hold_out = (rand_vals < hold_out_ratio) & og_station
    # hold_out_pos: og_station에서 hold_out 제외
    hold_out_pos = og_station & (~hold_out)
    # drop_out: hold_out_pos 중 25% 선택
    drop_out = (np.random.rand(*field.shape) < drop_out_ratio) & hold_out_pos
    return og_station, hold_out, hold_out_pos, drop_out

def plot_results(ds_original: xr.Dataset, ds_kriged: xr.Dataset, var_name: str, output_path: str):
    """
    ds_original: 원본 dataset
    ds_kriged: kriging 적용된 dataset
    var_name: 시각화할 변수 이름 (time, lat, lon 구조라고 가정)
    output_path: 저장할 그림 파일 경로
    """
    # 첫 두 시간(time index 0, 1)
    field0_orig = ds_original[var_name].isel(time=0).values  # (lat, lon)
    field1_orig = ds_original[var_name].isel(time=1).values
    field0_kriged = ds_kriged[var_name].isel(time=0).values
    field1_kriged = ds_kriged[var_name].isel(time=1).values

    # 각 시간별로 mask 계산 (seed를 달리하여 drop_out1과 drop_out2가 다르게)
    # 첫번째 시간: drop_out1
    _, hold_out0, hold_out_pos0, drop_out1 = compute_masks(field0_orig, hold_out_ratio=0.2, drop_out_ratio=0.25, seed=42)
    # 두번째 시간: drop_out2
    _, hold_out1, hold_out_pos1, drop_out2 = compute_masks(field1_orig, hold_out_ratio=0.2, drop_out_ratio=0.25, seed=99)

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    # --- 첫 번째 시간 ---
    # 그림 1-1: 원본
    im0 = axs[0,0].imshow(field0_orig, cmap='viridis')
    axs[0,0].set_title("original")
    plt.colorbar(im0, ax=axs[0,0])
    # 그림 1-2: kriging 결과
    im1 = axs[0,1].imshow(field0_kriged, cmap='viridis')
    axs[0,1].set_title("kriging ")
    plt.colorbar(im1, ax=axs[0,1])
    # 그림 1-3: 원본 위에 hold out (빨간점)
    im2 = axs[0,2].imshow(field0_orig, cmap='viridis')
    hold_y0, hold_x0 = np.where(hold_out0)  # 수정: hold_out0 사용
    axs[0,2].scatter(hold_x0, hold_y0, s=3, c='red', marker='o')
    axs[0,2].set_title(" hold out")
    plt.colorbar(im2, ax=axs[0,2])
    # 그림 1-4: 원본 위에 hold out (빨간점) + drop_out1 (파란점)
    im3 = axs[0,3].imshow(field0_orig, cmap='viridis')
    axs[0,3].scatter(hold_x0, hold_y0, s=3, c='red', marker='o')
    drop_y1, drop_x1 = np.where(drop_out1)
    axs[0,3].scatter(drop_x1, drop_y1, s=3, c='blue', marker='o')
    axs[0,3].set_title("hold out  + drop_out ")
    plt.colorbar(im3, ax=axs[0,3])

    # --- 두 번째 시간 ---
    # 그림 2-1: 원본
    im4 = axs[1,0].imshow(field1_orig, cmap='viridis')
    axs[1,0].set_title("origianl")
    plt.colorbar(im4, ax=axs[1,0])
    # 그림 2-2: kriging 결과
    im5 = axs[1,1].imshow(field1_kriged, cmap='viridis')
    axs[1,1].set_title(" kriging ")
    plt.colorbar(im5, ax=axs[1,1])
    # 그림 2-3: 원본 위에 hold out (빨간점)
    im6 = axs[1,2].imshow(field1_orig, cmap='viridis')
    hold_y1, hold_x1 = np.where(hold_out1)
    axs[1,2].scatter(hold_x1, hold_y1, s=3, c='red', marker='o')
    axs[1,2].set_title(" hold out")
    plt.colorbar(im6, ax=axs[1,2])
    # 그림 2-4: 원본 위에 hold out (빨간점) + drop_out2 (파란점)
    im7 = axs[1,3].imshow(field1_orig, cmap='viridis')
    axs[1,3].scatter(hold_x1, hold_y1, s=3, c='red', marker='o')
    drop_y2, drop_x2 = np.where(drop_out2)
    axs[1,3].scatter(drop_x2, drop_y2, s=3, c='blue', marker='o')
    axs[1,3].set_title("hold out  + drop_out")
    plt.colorbar(im7, ax=axs[1,3])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

if __name__ == "__main__":
    # 데이터셋 로드 (사용할 dataset 경로에 맞게 수정)
    ds = xr.open_dataset('/home/heo0614/temp_data/Weather/new_input_final/sparse_input2.nc')
    
    # Optional: customize kriging parameters
    config = KrigingConfig(
        variogram_model='exponential',
        sampling_ratio=0.025,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        exclude_vars=['land_sea_mask']
    )
    
    # kriging 적용된 dataset
    ds_kriged = dataset_to_array_kriging(ds, config)
    
    # 시각화에 사용할 변수 선택 (예시로 첫 번째 data_var 사용)
    var_name = list(ds.data_vars)[0]
    
    # 그림 저장
    output_path = "./kriging_visualization.png"
    plot_results(ds, ds_kriged, var_name, output_path)
    print(f"Visualization saved to {output_path}")

# %%
