import os
import copy
import time
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from kriging.kriging_interpolation import dataset_to_array_kriging, KrigingConfig
from metnet3_original import MetNet3

# --------------------------------------------------------------------------------
# Kriging 설정
# --------------------------------------------------------------------------------
config = KrigingConfig(
    variogram_model='exponential',
    sampling_ratio=0.025,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    exclude_vars=['land_sea_mask']  # 필요에 따라 제외할 변수 지정
)

# --------------------------------------------------------------------------------
# 전처리 함수
# --------------------------------------------------------------------------------
def process_dataarray(da: xr.DataArray, method="norm"):
    new_da = da.copy(deep=True)
    has_time = "time" in da.dims
    has_level = "level" in da.dims

    if has_level:
        for lev in da.coords["level"].values:
            slice_da = da.sel(level=lev)
            arr = slice_da.values
            if method == "precip":
                processed = np.tanh(np.log(arr + 1) / 4)
            else:
                mean = np.mean(arr)
                std = np.std(arr) + 1e-6
                processed = (arr - mean) / std
            new_da.loc[dict(level=lev)] = processed
    elif has_time:
        arr = da.values
        if method == "precip":
            processed = np.tanh(np.log(arr + 1) / 4)
        else:
            mean = np.mean(arr)
            std = np.std(arr) + 1e-6
            processed = (arr - mean) / std
        new_da[:] = processed
    else:
        arr = da.values
        if method == "precip":
            processed = np.tanh(np.log(arr + 1) / 4)
        else:
            mean = np.mean(arr)
            std = np.std(arr) + 1e-6
            processed = (arr - mean) / std
        new_da[:] = processed

    return new_da

def discretize_dataarray(da: xr.DataArray, num_bins: int):
    new_da = da.copy(deep=True)
    has_time = "time" in da.dims
    has_level = "level" in da.dims

    if has_level:
        for lev in da.coords["level"].values:
            slice_da = da.sel(level=lev)
            arr = slice_da.values
            min_val = np.min(arr)
            max_val = np.max(arr)
            bins = np.linspace(min_val, max_val, num_bins + 1)
            disc = np.digitize(arr, bins) - 1
            disc = np.clip(disc, 0, num_bins - 1)
            new_da.loc[dict(level=lev)] = disc
    elif has_time:
        arr = da.values
        min_val = np.min(arr)
        max_val = np.max(arr)
        bins = np.linspace(min_val, max_val, num_bins + 1)
        disc = np.digitize(arr, bins) - 1
        disc = np.clip(disc, 0, num_bins - 1)
        new_da[:] = disc
    else:
        arr = da.values
        min_val = np.min(arr)
        max_val = np.max(arr)
        bins = np.linspace(min_val, max_val, num_bins + 1)
        disc = np.digitize(arr, bins) - 1
        disc = np.clip(disc, 0, num_bins - 1)
        new_da[:] = disc

    return new_da

def process_dataset_inputs(ds: xr.Dataset) -> xr.Dataset:
    new_vars = {}
    for var in ds.data_vars:
        da = ds[var]
        if ("latitude" not in da.dims) or ("longitude" not in da.dims):
            new_vars[var] = da
        else:
            if var.lower() == "total_precipitation":
                new_vars[var] = process_dataarray(da, method="precip")
            else:
                new_vars[var] = process_dataarray(da, method="norm")
    return xr.Dataset(new_vars, coords=ds.coords)

def process_dataset_target_high(ds: xr.Dataset) -> xr.Dataset:
    new_vars = {}
    for var in ds.data_vars:
        da = ds[var]
        if ("latitude" not in da.dims) or ("longitude" not in da.dims):
            new_vars[var] = da
        else:
            if var.lower() == "total_precipitation":
                new_vars[var] = discretize_dataarray(da, 512)
            else:
                new_vars[var] = da
    return xr.Dataset(new_vars, coords=ds.coords)

def process_dataset_target_sparse(ds: xr.Dataset) -> xr.Dataset:
    new_vars = {}
    for var in ds.data_vars:
        da = ds[var]
        if ("latitude" not in da.dims) or ("longitude" not in da.dims):
            new_vars[var] = da
        else:
            if var.lower() in ["2m_dewpoint_temperature", "2m_temperature"]:
                new_vars[var] = discretize_dataarray(da, 256)
            else:
                new_vars[var] = da
    return xr.Dataset(new_vars, coords=ds.coords)

def process_dataset_target_dense(ds: xr.Dataset) -> xr.Dataset:
    new_vars = {}
    for var in ds.data_vars:
        da = ds[var]
        if ("latitude" not in da.dims) or ("longitude" not in da.dims):
            new_vars[var] = da
        else:
            new_vars[var] = process_dataarray(da, method="norm")
    return xr.Dataset(new_vars, coords=ds.coords)

def hold_out_input(ds_input: xr.Dataset, ds_target: xr.Dataset):
    """
    ds_input에서 20% 위치를 홀드아웃하고(학습 입력에서 제거),
    ds_target에서도 동일한 위치를 마스킹(crop에 맞춰서) 처리.
    """
    lat_dim = ds_input.dims.get('latitude')
    lon_dim = ds_input.dims.get('longitude')
    if lat_dim is None or lon_dim is None:
        raise ValueError("Dataset에 'latitude'와 'longitude' 좌표가 필요합니다.")

    # 1) ds_input 중 NaN이 아닌 위치(og_station) 찾기
    og_station = None
    for var in ds_input.data_vars:
        if 'latitude' in ds_input[var].dims and 'longitude' in ds_input[var].dims:
            idx = {dim: 0 for dim in ds_input[var].dims if dim not in ['latitude', 'longitude']}
            og_station = ds_input[var].isel(**idx).notnull()
            break
    if og_station is None:
        raise ValueError("ds_input에서 'latitude'와 'longitude' 차원을 가진 변수를 찾을 수 없습니다.")

    np.random.seed(42)
    random_mask = np.random.rand(*og_station.shape) < 0.2
    hold_out_20_mask = random_mask & og_station.values

    mask_da = xr.DataArray(
        hold_out_20_mask,
        dims=['latitude', 'longitude'],
        coords={'latitude': ds_input.latitude, 'longitude': ds_input.longitude}
    )

    # ds_input에서 해당 위치를 NaN 처리
    ds_masked = ds_input.copy()
    for var in ds_masked.data_vars:
        if 'latitude' in ds_masked[var].dims and 'longitude' in ds_masked[var].dims:
            ds_masked[var] = ds_masked[var].where(~mask_da)

    # ds_target도 같은 위치를 마스킹 (단, crop된 shape 고려)
    target_lat = ds_target.dims.get('latitude')
    target_lon = ds_target.dims.get('longitude')
    if target_lat is None or target_lon is None:
        raise ValueError("sparse_target 데이터셋에 'latitude'와 'longitude' 좌표가 필요합니다.")

    row_offset = (lat_dim - target_lat) // 2
    col_offset = (lon_dim - target_lon) // 2

    hold_out_20_mask_crop = hold_out_20_mask[row_offset:row_offset + target_lat,
                                             col_offset:col_offset + target_lon]
    mask_target = xr.DataArray(
        hold_out_20_mask_crop,
        dims=['latitude', 'longitude'],
        coords={'latitude': ds_target.latitude, 'longitude': ds_target.longitude}
    )
    ds_target_masked = ds_target.copy()
    for var in ds_target_masked.data_vars:
        if 'latitude' in ds_target_masked[var].dims and 'longitude' in ds_target_masked[var].dims:
            ds_target_masked[var] = ds_target_masked[var].where(~mask_target)

    return ds_masked, ds_target_masked, mask_da, og_station

# --------------------------------------------------------------------------------
# CRPS, CSI 계산 함수
# --------------------------------------------------------------------------------
def compute_crps(pred_probs: torch.Tensor, target_labels: torch.Tensor) -> float:
    """
    pred_probs: (B, C, H, W) - softmax 확률분포
    target_labels: (B, H, W) - 0~C-1 레이블
    """
    B, C, H, W = pred_probs.shape
    cdf_pred = torch.cumsum(pred_probs, dim=1)
    device = target_labels.device
    cdf_true = (torch.arange(C, device=device).view(1, C, 1, 1) >= target_labels.unsqueeze(1)).float()
    diff_sq = (cdf_pred - cdf_true) ** 2
    crps_val = diff_sq.mean().item()
    return crps_val

def compute_csi(pred_probs: torch.Tensor, target_labels: torch.Tensor,
                threshold_bin: int = 8, prob_threshold: float = 0.5) -> float:
    """
    threshold_bin 이상의 bin 확률 합이 prob_threshold 이상이면 '강수'로 간주
    """
    B, C, H, W = pred_probs.shape
    p_rain = pred_probs[:, threshold_bin:, :, :].sum(dim=1)  # (B,H,W)
    pred_positive = (p_rain >= prob_threshold)
    real_positive = (target_labels >= threshold_bin)

    tp = (pred_positive & real_positive).sum().item()
    fp = (pred_positive & ~real_positive).sum().item()
    fn = (~pred_positive & real_positive).sum().item()
    denom = tp + fp + fn
    csi = tp / denom if denom > 0 else 0.0
    return csi

# --------------------------------------------------------------------------------
# WeatherBenchNetCDFDataset
# --------------------------------------------------------------------------------
class WeatherBenchNetCDFDataset(Dataset):
    def __init__(self, root_dir: str):
        """
        새 NetCDF 파일들을 열고, time 인덱스를 기준으로
        t..t+5 => input, t+6..t+11 => target
        """
        self.root_dir = root_dir

        # -----------------------------
        # 1) NetCDF 불러오기
        # -----------------------------
        sparse_input_path  = os.path.join(root_dir, 'sparse_input2.nc')
        dense_input_path   = os.path.join(root_dir, 'dense_input.nc')
        low_inputs_path    = os.path.join(root_dir, 'low_inputs.nc')
        sparse_target_path = os.path.join(root_dir, 'sparse_target2.nc')
        dense_target_path  = os.path.join(root_dir, 'dense_target.nc')
        high_target_path   = os.path.join(root_dir, 'high_target.nc')

        self.ds_sparse_input  = xr.open_dataset(sparse_input_path).load()
        self.ds_dense_input   = xr.open_dataset(dense_input_path).load()
        self.ds_low_inputs    = xr.open_dataset(low_inputs_path).load()
        self.ds_sparse_target = xr.open_dataset(sparse_target_path).load()
        self.ds_dense_target  = xr.open_dataset(dense_target_path).load()
        self.ds_high_target   = xr.open_dataset(high_target_path).load()

        # 2) 20% hold-out 적용 -> 80%만 남긴 mask
        self.ds_sparse_input, self.ds_sparse_target, self.mask_da, self.og_station = hold_out_input(
            self.ds_sparse_input, self.ds_sparse_target
        )

        # 3) 크리깅으로 보간
        start_time = time.time()
        self.ds_sparse_input = dataset_to_array_kriging(self.ds_sparse_input, config)
        self.ds_sparse_target = dataset_to_array_kriging(self.ds_sparse_target, config)
        elapsed = time.time() - start_time
        print(f"Kriging interpolation complete. Elapsed time: {elapsed:.2f} seconds.")

        # 4) 전처리
        self.ds_sparse_input  = process_dataset_inputs(self.ds_sparse_input)
        self.ds_dense_input   = process_dataset_inputs(self.ds_dense_input)
        self.ds_low_inputs    = process_dataset_inputs(self.ds_low_inputs)

        self.ds_high_target   = process_dataset_target_high(self.ds_high_target)
        self.ds_sparse_target = process_dataset_target_sparse(self.ds_sparse_target)
        self.ds_dense_target  = process_dataset_target_dense(self.ds_dense_target)

        # 5) time 슬라이싱 위한 설정
        self.total_time = self.ds_sparse_input.dims['time']
        self.max_t = self.total_time - 12
        if self.max_t <= 0:
            raise ValueError("time 차원이 12보다 작아 슬라이딩 불가합니다.")

        # 6) 변수 목록
        self.sparse_vars = list(self.ds_sparse_input.data_vars.keys())
        self.dense_vars  = list(self.ds_dense_input.data_vars.keys())
        self.low_vars    = list(self.ds_low_inputs.data_vars.keys())

        self.sparse_target_vars = list(self.ds_sparse_target.data_vars.keys())
        self.dense_target_vars  = list(self.ds_dense_target.data_vars.keys())
        self.high_target_vars   = list(self.ds_high_target.data_vars.keys())  # ['total_precipitation']

    def __len__(self):
        return self.max_t + 1

    def __getitem__(self, idx: int):
        t_start = idx
        t_end   = idx + 5
        tgt_start = idx + 6
        tgt_end   = idx + 11

        # =========================================================
        # (1) Sparse Input (36,156,156) 예시
        # =========================================================
        sparse_list = []
        for var_name in self.sparse_vars:
            data_6h = self.ds_sparse_input[var_name][t_start:t_end+1, ...]  # (6, lat, lon)
            sparse_list.append(data_6h.values)
        input_sparse_np = np.concatenate(sparse_list, axis=0)  # (6*num_vars, lat, lon)

        # ---------------------------------------------------------
        # * Densification *
        #  - og_station & ~mask_da => 학습에 쓸 80% 위치
        #  - 해당 위치 중 25% 랜덤 드롭 -> 입력 0으로 설정
        # ---------------------------------------------------------
        # og_station / mask_da : shape (lat, lon)
        valid_mask = self.og_station.values & (~self.mask_da.values)  # True인 곳이 '80%'
        dropout_mask = (np.random.rand(*valid_mask.shape) < 0.25) & valid_mask
        # 모든 채널에 대해 dropout_mask 위치를 0으로
        input_sparse_np[:, dropout_mask] = 0.0

        input_sparse = torch.from_numpy(input_sparse_np).float()  # (C, 156, 156)

        # =========================================================
        # (2) Dense Input
        # =========================================================
        dense_5var_list = []
        total_precip_6h = None
        land_sea_mask_2d = None

        for var_name in self.dense_vars:
            da = self.ds_dense_input[var_name]
            dims = da.dims
            if ('time' not in dims) and ('level' not in dims):
                land_sea_mask_2d = da.values[None, ...]
            elif ('time' in dims) and ('level' not in dims):
                tp_6h = da[t_start:t_end+1].values
                total_precip_6h = tp_6h
            else:
                sliced = da[t_start:t_end+1, ...].values  # (6, level, lat, lon)
                lvl_size = sliced.shape[1]
                reshaped = sliced.reshape(-1, sliced.shape[2], sliced.shape[3])
                dense_5var_list.append(reshaped)

        dense_5var_cat = np.concatenate(dense_5var_list, axis=0) if len(dense_5var_list) > 0 else np.empty((0,))
        if total_precip_6h is None:
            raise ValueError("dense_input에서 total_precipitation(time, lat, lon) 변수를 찾지 못했습니다.")
        if land_sea_mask_2d is None:
            raise ValueError("dense_input에서 land_sea_mask 변수를 찾지 못했습니다.")

        dense_input_np = np.concatenate([dense_5var_cat, total_precip_6h, land_sea_mask_2d], axis=0)
        input_dense = torch.from_numpy(dense_input_np).float()

        # =========================================================
        # (3) input_stale = dense_input의 total_precip 6시간
        # =========================================================
        input_stale_np = total_precip_6h
        input_stale = torch.from_numpy(input_stale_np).float()  # (6,156,156)

        # =========================================================
        # (4) input_low
        # =========================================================
        low_list = []
        for var_name in self.low_vars:
            da = self.ds_low_inputs[var_name]
            low_6h = da[t_start:t_end+1].values  # (6, lat, lon)
            low_list.append(low_6h)
        input_low_np = np.concatenate(low_list, axis=0)  # (6*num_vars, lat, lon)
        input_low = torch.from_numpy(input_low_np).float()

        # =========================================================
        # (5) 타겟들
        # =========================================================
        # (A) high_target => total_precip (512 bins)
        high_da = self.ds_high_target[self.high_target_vars[0]]
        high_6h_np = high_da[tgt_start:tgt_end+1].values  # (6,128,128)
        high_precip_6h = torch.from_numpy(high_6h_np).long()
        precipitation_target = high_precip_6h[-1]  # (128,128)

        # (B) sparse_target => 2m_temp, 2m_dewpoint, total_precip(하지만 여기서는 별도 사용x)
        s_t2m_da = self.ds_sparse_target['2m_temperature'][tgt_start:tgt_end+1].values  # (6,32,32)
        s_d2m_da = self.ds_sparse_target['2m_dewpoint_temperature'][tgt_start:tgt_end+1].values  # (6,32,32)
        s_tp_da  = self.ds_sparse_target['total_precipitation'][tgt_start:tgt_end+1].values  # (6,32,32)

        t2m_6h    = torch.from_numpy(s_t2m_da).long()
        d2m_6h    = torch.from_numpy(s_d2m_da).long()
        # sparse_target의 precipitation은 현재 예시에서는 사용하지 않고 있음
        # 필요하다면 별도 로스에 활용 가능

        surface_targets = {
            "temperature_2m": t2m_6h[-1],  # (32,32)
            "dewpoint_2m":    d2m_6h[-1],  # (32,32)
        }

        # (C) dense_target => (5개 var: time+level) + (total_precip) + (mask)
        dense_5var_list_t = []
        total_precip_6h_t = None
        land_sea_mask_2d_t = None

        for var_name in self.dense_target_vars:
            da = self.ds_dense_target[var_name]
            dims = da.dims
            if ('time' not in dims) and ('level' not in dims):
                land_sea_mask_2d_t = da.values[None, ...]
            elif ('time' in dims) and ('level' not in dims):
                tp_6h = da[tgt_start:tgt_end+1].values  # (6, 32, 32)
                total_precip_6h_t = tp_6h
            else:
                sliced = da[tgt_start:tgt_end+1, ...].values  # (6, level, 32, 32)
                lvl_size = sliced.shape[1]
                reshaped = sliced.reshape(-1, sliced.shape[2], sliced.shape[3])
                dense_5var_list_t.append(reshaped)

        dense_5var_cat_t = np.concatenate(dense_5var_list_t, axis=0) if len(dense_5var_list_t) > 0 else np.empty((0,))
        if total_precip_6h_t is None:
            raise ValueError("dense_target에서 total_precipitation(time, lat, lon)을 찾지 못했습니다.")
        if land_sea_mask_2d_t is None:
            raise ValueError("dense_target에서 land_sea_mask를 찾지 못했습니다.")

        dense_target_np = np.concatenate([dense_5var_cat_t, total_precip_6h_t, land_sea_mask_2d_t], axis=0)
        hrrr_target = torch.from_numpy(dense_target_np).float()

        lead_time = torch.tensor(np.random.randint(0, 722), dtype=torch.long)

        sample = {
            "lead_time": lead_time,
            "input_sparse": input_sparse,   # (C, 156, 156)
            "input_stale":  input_stale,   # (6, 156, 156)
            "input_dense":  input_dense,   # (ch, 156, 156)
            "input_low":    input_low,     # (ch, 156, 156)
            "precipitation_targets": {
                "total_precipitation": precipitation_target  # (128,128)
            },
            "surface_targets": surface_targets,   # dict of 2D (32,32)
            "hrrr_target": hrrr_target,           # (ch, 32,32)
        }
        return sample

# --------------------------------------------------------------------------------
# Dataset split
# --------------------------------------------------------------------------------
def create_datasets(root_dir):
    dataset = WeatherBenchNetCDFDataset(root_dir)
    N = len(dataset)
    train_end = int(0.50 * N)
    val_end   = int(0.75 * N)
    indices = np.arange(N)

    train_dataset = Subset(dataset, indices[:train_end])
    val_dataset   = Subset(dataset, indices[train_end:val_end])
    test_dataset  = Subset(dataset, indices[val_end:])
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(root_dir, batch_size=2, num_workers=2):
    train_dataset, val_dataset, test_dataset = create_datasets(root_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

# --------------------------------------------------------------------------------
# 2-Stage Training (Stage1 -> Stage2)
# --------------------------------------------------------------------------------

def train_one_stage(
    metnet3,
    train_loader,
    val_loader,
    device,
    stage_idx=1,
    surface_loss_mult=1.0,
    min_epochs=30,
    max_no_improve=5,
    max_epochs=120,
    log_dir="/home/heo0614/temp_data/Weather/test"
):
    """
    stage_idx: 1 or 2
    surface_loss_mult: surface 로스에 곱할 가중치 (Stage2에서 100으로 설정)
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"training_log_stage{stage_idx}2.txt")
    model_path = os.path.join(log_dir, f"metnet3_stage{stage_idx}2.pth")

    optimizer = torch.optim.Adam(metnet3.parameters(), lr=1e-4)

    best_crps = float('inf')
    best_csi  = 0.0
    best_model_state = None

    no_improve_count = 0
    epoch = 0

    with open(log_path, 'w') as f_log:  # 오픈해두고 기록
        while True:
            epoch += 1
            metnet3.train()
            epoch_loss = 0.0

            for batch in train_loader:
                lead_times = batch['lead_time'].to(device)
                in_sparse  = batch['input_sparse'].to(device)
                in_stale   = batch['input_stale'].to(device)
                in_dense   = batch['input_dense'].to(device)
                in_low     = batch['input_low'].to(device)

                precip_targets = { k: v.to(device) for k,v in batch['precipitation_targets'].items() }
                surface_targets = { k: v.to(device) for k,v in batch['surface_targets'].items() }
                hrrr_target = batch['hrrr_target'].to(device)

                optimizer.zero_grad()
                total_loss, loss_breakdown = metnet3(
                    lead_times            = lead_times,
                    hrrr_input_2496       = in_dense,
                    hrrr_stale_state      = in_stale,
                    input_2496            = in_sparse,
                    input_4996            = in_low,
                    precipitation_targets = precip_targets,
                    surface_targets       = surface_targets,
                    hrrr_target           = hrrr_target,
                )

                # -------------------------------------------
                # Stage2에서는 surface 로스를 100배로 가중
                # -------------------------------------------
                # 메타넷 코드에서 total_loss = sum(각 로스 * 내부 가중) 으로 계산되었다고 가정
                # 우리는 breakdown에서 surface 로스 부분을 찾아서 조정
                if 'surface' in loss_breakdown:  # 예: loss_breakdown['surface']
                    surface_loss = loss_breakdown['surface']
                    total_loss = total_loss - surface_loss + surface_loss_mult * surface_loss

                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)

            # -----------------
            # Validation
            # -----------------
            metnet3.eval()
            total_crps_val = 0.0
            total_csi_val = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    lead_times = batch['lead_time'].to(device)
                    in_sparse  = batch['input_sparse'].to(device)
                    in_stale   = batch['input_stale'].to(device)
                    in_dense   = batch['input_dense'].to(device)
                    in_low     = batch['input_low'].to(device)

                    precip_targets = { k: v.to(device) for k,v in batch['precipitation_targets'].items() }
                    # surface_targets = { k: v.to(device) for k,v in batch['surface_targets'].items() } # CSI/CRPS는 precipitation만 예시

                    pred = metnet3(
                        lead_times       = lead_times,
                        hrrr_input_2496  = in_dense,
                        hrrr_stale_state = in_stale,
                        input_2496       = in_sparse,
                        input_4996       = in_low,
                    )
                    precipitation_preds = pred.precipitation
                    logits = precipitation_preds['total_precipitation']  # (B, C, H, W)
                    probs = F.softmax(logits, dim=1)
                    tgt = precip_targets['total_precipitation']
                    crps_batch = compute_crps(probs, tgt)
                    csi_batch  = compute_csi(probs, tgt, threshold_bin=8, prob_threshold=0.5)

                    total_crps_val += crps_batch
                    total_csi_val  += csi_batch
                    num_batches += 1

            mean_crps = total_crps_val / num_batches
            mean_csi  = total_csi_val / num_batches

            # 로그 기록
            log_line = (f"[Stage {stage_idx} | Epoch {epoch}] "
                        f"TrainLoss={avg_epoch_loss:.4f}, CRPS={mean_crps:.4f}, CSI={mean_csi:.4f}\n")
            print(log_line.strip())
            f_log.write(log_line)
            f_log.flush()

            # Early stopping 체크
            improved = False
            if mean_crps < best_crps:
                best_crps = mean_crps
                improved = True
            if mean_csi > best_csi:
                best_csi = mean_csi
                improved = True

            if improved:
                best_model_state = copy.deepcopy(metnet3.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            if (epoch >= min_epochs and no_improve_count >= max_no_improve) or (epoch >= max_epochs):
                stop_line = f"Stage {stage_idx} stopped at epoch {epoch} (no_improve={no_improve_count}).\n"
                print(stop_line.strip())
                f_log.write(stop_line)
                break

    # Stage 종료 후 Best 모델 저장
    if best_model_state is not None:
        metnet3.load_state_dict(best_model_state)

    torch.save(metnet3.state_dict(), model_path)
    print(f"Stage {stage_idx} best model saved to {model_path}")

def main():
    root_dir = "/home/heo0614/temp_data/Weather/new_input_final"
    train_loader, val_loader, test_loader = create_dataloaders(root_dir, batch_size=2, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------
    # 모델 초기화
    # -----------------------------------------
    metnet3 = MetNet3(
        dim = 512,
        num_lead_times = 722,
        lead_time_embed_dim = 32,
        input_spatial_size = 156,
        attn_depth = 12,
        attn_dim_head = 8,
        attn_heads = 32,
        attn_dropout = 0.1,
        vit_window_size = 8,
        vit_mbconv_expansion_rate = 4,
        vit_mbconv_shrinkage_rate = 0.25,
        hrrr_channels = 157,  
        input_2496_channels = 36,
        input_4996_channels = 12,
        surface_and_hrrr_target_spatial_size = 32,
        precipitation_target_bins = dict(
            total_precipitation = 512,
        ),
        surface_target_bins = dict(
            temperature_2m = 256,
            dewpoint_2m    = 256,
        ),
        hrrr_loss_weight = 10,
        hrrr_norm_strategy = 'sync_batchnorm',
        hrrr_norm_statistics = None,
        crop_size_post_16km = 32,
        resnet_block_depth = 2,
    )
    metnet3.to(device)

    # -----------------------------------------
    # Stage 1: 강우량(High target) 중심이지만 모든 target 학습
    # -----------------------------------------
    train_one_stage(
        metnet3,
        train_loader,
        val_loader,
        device,
        stage_idx=1,
        surface_loss_mult=1.0,  # 기본 가중치
        min_epochs=30,
        max_no_improve=5,
        max_epochs=120,
        log_dir="/home/heo0614/temp_data/Weather/test"
    )

    # -----------------------------------------
    # Stage 2: Sparse 데이터(서피스) 가중치 100배로 fine-tuning
    #          (모든 타겟 학습, 단 surface_loss만 100배)
    # -----------------------------------------
    train_one_stage(
        metnet3,
        train_loader,
        val_loader,
        device,
        stage_idx=2,
        surface_loss_mult=100.0,
        min_epochs=30,
        max_no_improve=5,
        max_epochs=120,
        log_dir="/home/heo0614/temp_data/Weather/test"
    )

    print("Training complete. Both stages finished.")


if __name__ == "__main__":
    main()
