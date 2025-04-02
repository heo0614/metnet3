# %% 

import os
import xarray as xr
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------------------------
# (A) 경로 및 폴더 설정
# -------------------------------------------------
PATH_SPARSE_INPUT = r"/projects/aiid/KIPOT_SKT/Weather/sparse_data_input/156x156_sparse_0.5_input_all_80_interp.nc"
PATH_DENSE_INPUT  = r"/projects/aiid/KIPOT_SKT/Weather/dense_data_input/156x156_dense_0.5_input_all.nc"
PATH_LOW_INPUT    = r"/projects/aiid/KIPOT_SKT/Weather/low_data_input/156x156_low_1.0_input_all.nc"

PATH_HIGH_TARGET   = r"/projects/aiid/KIPOT_SKT/Weather/high_data_target/128x128_high_target_0.25_all.nc"
PATH_SPARSE_TARGET = r"/projects/aiid/KIPOT_SKT/Weather/sparse_data_target/32x32_sparse_target_0.5_all.nc"
PATH_DENSE_TARGET  = r"/projects/aiid/KIPOT_SKT/Weather/dense_data_target/32x32_dense_target_0.5_all.nc"

SAVE_DIR_TRAIN = r"/projects/aiid/KIPOT_SKT/Weather/trainset"
SAVE_DIR_VALID = r"/projects/aiid/KIPOT_SKT/Weather/validationset"
SAVE_DIR_TEST  = r"/projects/aiid/KIPOT_SKT/Weather/testset"

# [수정 사항]
# stale_input은 원본 파일 사용, sparse_input은 80% zero 처리된 파일 사용
SAVE_ZERO_PATH    = r"/projects/aiid/KIPOT_SKT/Weather/sparse_data_input/156x156_sparse_0.5_input_all_80_zero.nc"
# [변경] Sparse Target의 저장 경로 변경
SAVE_SPARSE_TARGET_ZERO = r"/projects/aiid/KIPOT_SKT/Weather/sparse_data_target/32x32_sparse_target_0.5_all_80_interp.nc"
SAVE_INDEX_PATH = r"/projects/aiid/KIPOT_SKT/Weather/sparse_data_input/selected_indices_80.npy"
def make_dirs_for_targets(base_dir, subfolders):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    for sf in subfolders:
        path_ = os.path.join(base_dir, sf)
        os.makedirs(path_, exist_ok=True)

# -------------------------------------------------
# (B) NetCDF 로드
# -----------------------                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          --------------------------
ds_sparse_stale = xr.open_dataset(PATH_SPARSE_INPUT)
ds_sparse_interp = xr.open_dataset(SAVE_ZERO_PATH)

ds_dense_input   = xr.open_dataset(PATH_DENSE_INPUT)
ds_low_input     = xr.open_dataset(PATH_LOW_INPUT)

ds_high_target   = xr.open_dataset(PATH_HIGH_TARGET)
ds_sparse_target = xr.open_dataset(PATH_SPARSE_TARGET)
ds_dense_target  = xr.open_dataset(PATH_DENSE_TARGET)

# -------------------------------------------------
# (C) 사용할 변수명들
# -------------------------------------------------
sparse_all_vars = list(ds_sparse_stale.data_vars.keys())
SPARSE_STALE_VAR = '2m_temperature'
sparse_input_vars = [v for v in sparse_all_vars if v != SPARSE_STALE_VAR]

dense_all_vars = list(ds_dense_input.data_vars.keys())
dense_input_vars = dense_all_vars

low_all_vars = list(ds_low_input.data_vars.keys())
low_input_vars = low_all_vars

sparse_target_vars = list(ds_sparse_target.data_vars.keys())
high_target_vars   = list(ds_high_target.data_vars.keys())
dense_target_vars  = list(ds_dense_target.data_vars.keys())

# -------------------------------------------------
# (E) 변수별 (min, max)와 bin 개수 설정
# -------------------------------------------------
variable_range_info = {
    "2m_temperature":          (240.0, 330.0),
    "2m_dewpoint_temperature": (235.0, 310.0),
    "surface_pressure":        (40000.0, 105000.0),
    "total_precipitation":     (0.0, 0.05),
    "u_component_of_wind":     (-30.0, 50.0),
    "v_component_of_wind":     (-25.0, 30.0),
    "geopotential":            (120000.0, 135000.0),
    "land_sea_mask":           (0.0, 1.0),
    "temperature":             (240.0, 280.0),
    "10m_u_component_of_wind": (-25.0, 25.0),
    "10m_v_component_of_wind": (-25.0, 25.0),
    "specific_humidity":       (0.0001, 0.01),
    "2m_temperature_target":           (280.0, 330.0),
    "2m_dewpoint_temperature_target":  (285.0, 305.0),
    "total_precipitation_target":      (0.0, 0.05),
    "u_component_of_wind_target":      (-30.0, 50.0),
    "v_component_of_wind_target":      (-25.0, 30.0),
}

target_bins = {
    "2m_temperature": 256,
    "2m_dewpoint_temperature": 256,
    "total_precipitation": 256,
    "surface_pressure": 256,
    "u_component_of_wind": 180,
    "v_component_of_wind": 180,
    "geopotential": 256,
    "land_sea_mask": 2,
    "temperature": 256,
    "10m_u_component_of_wind": 256,
    "10m_v_component_of_wind": 256,
    "specific_humidity": 256,
    "total_precipitation_target": 512,
    "total_cloud_cover": 256,
}

variable_norm_info = {
    "2m_temperature":          (240.0, 330.0),
    "2m_dewpoint_temperature": (235.0, 310.0),
    "u_component_of_wind":     (-30.0, 50.0),
    "v_component_of_wind":     (-25.0, 30.0),
    "geopotential":            (120000.0, 135000.0),
    "land_sea_mask":           (0.0, 1.0),
    "temperature":             (240.0, 280.0),
    "10m_u_component_of_wind": (-25.0, 25.0),
    "10m_v_component_of_wind": (-25.0, 25.0),
    "specific_humidity":       (0.0001, 0.01),
    "total_cloud_cover":       (0.0, 1.0),
}

# -------------------------------------------------
# (F) 정규화 함수
# -------------------------------------------------
def normalize_array(var_name: str, arr: np.ndarray) -> np.ndarray:
    if var_name == "total_precipitation":
        return log_tanh_transform(arr)
    if var_name in variable_norm_info:
        vmin, vmax = variable_norm_info[var_name]
    else:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if vmin == vmax:
            vmax = vmin + 1e-5
    normalized = (arr - np.mean(arr)) / np.std(arr)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)

def log_tanh_transform(arr: np.ndarray) -> np.ndarray:
    arr_clip = np.clip(arr, 0, None)
    val_log  = np.log1p(arr_clip) / 4.0
    val_tanh = np.tanh(val_log)
    val_norm = (val_tanh + 1.0) / 2.0
    return val_norm

# -------------------------------------------------
# (G) dataset -> (time, channel, lat, lon) 변환 함수
# -------------------------------------------------
def dataset_to_array(ds: xr.Dataset, var_list: list):
    arrays = []
    for var in var_list:
        data_3d = ds[var].values
        data_4d = data_3d[:, np.newaxis, :, :]
        arrays.append(data_4d)
    return np.concatenate(arrays, axis=1)

def dataset_to_array_interpolated(ds: xr.Dataset, var_list: list):
    arrays = []
    for var in var_list:
        da = ds[var]
        da = da.where(da != 0, np.nan)
        da = da.sortby("latitude")
        da = da.interpolate_na(dim="latitude", method="linear", fill_value="extrapolate")
        da = da.sortby("longitude")
        da = da.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")
        data_3d = da.values
        data_4d = data_3d[:, np.newaxis, :, :]
        arrays.append(data_4d)
    return np.concatenate(arrays, axis=1)

def make_input_array(full_array, start_idx, window_size=6):
    slice_ = full_array[start_idx : start_idx + window_size]
    slice_transposed = slice_.transpose(1, 0, 2, 3)
    return slice_transposed.reshape(-1, slice_transposed.shape[2], slice_transposed.shape[3])

# -------------------------------------------------
# (H) Save functions
# -------------------------------------------------
def save_input_list(folder, name, array_list, var_list):
    if not array_list:
        print(f"  -> {name} list is empty, skip saving.")
        return
    arr = np.concatenate(array_list, axis=0)
    window_size = 6
    normalized_channels = []
    for c_idx, var in enumerate(var_list):
        for w in range(window_size):
            channel_idx = c_idx * window_size + w
            channel_data = arr[:, channel_idx, :, :]
            normalized = normalize_array(var, channel_data)
            normalized_channels.append(normalized)
    normalized_arr = np.stack(normalized_channels, axis=1)
    out_path = os.path.join(folder, f"{name}_normalized.npy")
    np.save(out_path, normalized_arr)
    print(f"  -> Saved {name}_normalized: shape = {normalized_arr.shape} -> {out_path}")

def save_target_dict(folder, subfolder, target_dict, var_list):
    outdir = os.path.join(folder, subfolder)
    for var in var_list:
        arr_list = target_dict[var]
        if not arr_list:
            print(f"  -> {var} list is empty, skip.")
            continue
        arr_cat = np.concatenate(arr_list, axis=0)
        orig_shape = arr_cat.shape
        reshaped_2d = arr_cat.reshape(-1, orig_shape[-2], orig_shape[-1])
        scaled_2d = linear_scale_and_clamp_to_int(var, reshaped_2d, subfolder=subfolder)
        arr_scaled = scaled_2d.reshape(orig_shape)
        fpath = os.path.join(outdir, f"{var}.npy")
        np.save(fpath, arr_scaled)
        print(f"  -> Saved {subfolder}/{var}.npy : shape={arr_scaled.shape}")

def linear_scale_and_clamp_to_int(var_name: str, arr: np.ndarray, subfolder: str = None) -> np.ndarray:
    if subfolder == "sparse_target":
        # 0인 값은 그대로 유지하고, 나머지 값에 대해서만 scaling 적용
        nonzero = arr[arr != 0]
        if nonzero.size > 0:
            vmin = np.min(nonzero)
            vmax = np.max(nonzero)
        else:
            vmin, vmax = 0, 1
        nbins = target_bins.get(var_name, 256)
        arr_scaled = np.zeros_like(arr, dtype=np.float32)
        mask = (arr != 0)
        arr_scaled[mask] = (arr[mask] - vmin) / (vmax - vmin) * (nbins - 1)
    else:
        if subfolder == "high_target" and var_name == "total_precipitation":
            nbins = 512
        else:
            nbins = target_bins.get(var_name, 256)
        if var_name in variable_range_info:
            vmin, vmax = variable_range_info[var_name]
        else:
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
            if vmin == vmax:
                vmax = vmin + 1e-5
        arr_scaled = (arr - vmin) / (vmax - vmin) * (nbins - 1)
    np.clip(arr_scaled, 0, nbins - 1, out=arr_scaled)
    arr_scaled = np.round(arr_scaled).astype(np.int32)
    return arr_scaled

def save_dense_target_as_one_file(folder, arr_list, file_name="dense_target"):
    if not arr_list:
        print(f"  -> {file_name} list is empty, skip saving.")
        return
    arr_cat = np.concatenate(arr_list, axis=0)
    N, T, C, H, W = arr_cat.shape
    arr_cat = arr_cat.transpose(0, 2, 1, 3, 4)
    dense_var_order = [
        "geopotential", "land_sea_mask", "temperature", 
        "10m_u_component_of_wind", "10m_v_component_of_wind", "specific_humidity"
    ]
    for var_idx, var_name in enumerate(dense_var_order):
        ch_data = arr_cat[:, var_idx]
        arr_cat[:, var_idx] = normalize_dense_target(ch_data)
    arr_cat = arr_cat.reshape(N, C*T, H, W)
    out_path = os.path.join(folder, f"{file_name}.npy")
    np.save(out_path, arr_cat)
    print(f"  -> Saved {file_name}.npy (normalized) : shape = {arr_cat.shape}")

def normalize_dense_target(arr: np.ndarray) -> np.ndarray:
    return (arr - np.mean(arr)) / np.std(arr)

# -------------------------------------------------
# (I) Main Preprocessing
# -------------------------------------------------
def main():
    global n_x, n_y, mask_2d
    n_x, n_y = 156, 156
    total_points = n_x * n_y
    selected_indices_2d = np.load(SAVE_INDEX_PATH)
    mask_1d = np.zeros(total_points, dtype=bool)
    mask_1d[selected_indices_2d[:, 0] * n_y + selected_indices_2d[:, 1]] = True
    mask_2d = mask_1d.reshape(n_x, n_y)
    
    print("[1] 각 Dataset을 (time, channel, lat, lon) 형태로 변환합니다.")
    arr_stale_state = dataset_to_array(ds_sparse_stale, [SPARSE_STALE_VAR])
    arr_sparse_input = dataset_to_array_interpolated(ds_sparse_interp, sparse_input_vars)
    
    arr_dense_input = dataset_to_array(ds_dense_input, dense_input_vars)
    arr_low_input = dataset_to_array(ds_low_input, low_input_vars)
    
    arr_sparse_target = dataset_to_array(ds_sparse_target, sparse_target_vars)
    arr_high_target = dataset_to_array(ds_high_target, high_target_vars)
    arr_dense_target = dataset_to_array(ds_dense_target, dense_target_vars)
    
    TIME_WINDOW_INPUT = 6
    TIME_WINDOW_TARGET = 6
    total_time = arr_sparse_input.shape[0]
    max_start = total_time - (TIME_WINDOW_INPUT + TIME_WINDOW_TARGET)
    
    TRAIN_END = 720
    VALID_END = 1056
    TEST_END  = 1392
    
    subfolders = [
        "sparse_target", "dense_target", "high_target",
        "input_sparse", "input_stale", "input_dense", "input_low"
    ]
    make_dirs_for_targets(SAVE_DIR_TRAIN, subfolders)
    make_dirs_for_targets(SAVE_DIR_VALID, subfolders)
    make_dirs_for_targets(SAVE_DIR_TEST, subfolders)
    
    sparse_input_train = []
    sparse_input_valid = []
    sparse_input_test  = []
    
    stale_state_train  = []
    stale_state_valid  = []
    stale_state_test   = []
    
    dense_input_train = []
    dense_input_valid = []
    dense_input_test  = []
    
    low_input_train   = []
    low_input_valid   = []
    low_input_test    = []
    
    sparse_target_train_dict = {v: [] for v in sparse_target_vars}
    sparse_target_valid_dict = {v: [] for v in sparse_target_vars}
    sparse_target_test_dict  = {v: [] for v in sparse_target_vars}
    
    high_target_train_dict   = {v: [] for v in high_target_vars}
    high_target_valid_dict   = {v: [] for v in high_target_vars}
    high_target_test_dict    = {v: [] for v in high_target_vars}
    
    dense_target_train = []
    dense_target_valid = []
    dense_target_test  = []
    
    print(f"[2] 슬라이딩 윈도우 진행: 총 {max_start+1}개 샘플 예상.")
    for start_idx in tqdm(range(max_start + 1)):
        target_start = start_idx + TIME_WINDOW_INPUT
        target_end   = target_start + TIME_WINDOW_TARGET
        
        if target_start <= TRAIN_END:
            split = "train"
        elif TRAIN_END < target_start <= VALID_END:
            split = "valid"
        elif VALID_END < target_start <= TEST_END:
            split = "test"
        else:
            continue
        
        si = make_input_array(arr_sparse_input, start_idx, TIME_WINDOW_INPUT)
        st = make_input_array(arr_stale_state, start_idx, TIME_WINDOW_INPUT)
        di = make_input_array(arr_dense_input, start_idx, TIME_WINDOW_INPUT)
        li = make_input_array(arr_low_input, start_idx, TIME_WINDOW_INPUT)
        
        if split == "train":
            sparse_input_train.append(si[None, ...])
            stale_state_train.append(st[None, ...])
            dense_input_train.append(di[None, ...])
            low_input_train.append(li[None, ...])
        elif split == "valid":
            sparse_input_valid.append(si[None, ...])
            stale_state_valid.append(st[None, ...])
            dense_input_valid.append(di[None, ...])
            low_input_valid.append(li[None, ...])
        else:
            sparse_input_test.append(si[None, ...])
            stale_state_test.append(st[None, ...])
            dense_input_test.append(di[None, ...])
            low_input_test.append(li[None, ...])
        
        # Sparse target 처리:
        # arr_sparse_target의 shape: (T, num_vars, 32, 32)
        target_nx, target_ny = 32, 32
        scale_factor = n_x / target_nx
        target_mask = np.zeros((target_nx, target_ny), dtype=bool)
        for i in range(target_nx):
            for j in range(target_ny):
                x_in = int((i + 0.5) * scale_factor)
                y_in = int((j + 0.5) * scale_factor)
                x_in = min(x_in, n_x - 1)
                y_in = min(y_in, n_y - 1)
                target_mask[i, j] = mask_2d[x_in, y_in]
        
        st_full = arr_sparse_target[target_start:target_end]  # (T, num_vars, 32, 32)
        # [변경 사항]
        # target_mask가 True인 영역에 대해서만 0으로 치환하고, 0이 아닌 값은 그대로 유지.
        # 조건 배열을 (1,1,32,32)로 확장하여 st_full에 브로드캐스트합니다.
        st_full_processed = np.where(target_mask[np.newaxis, np.newaxis, :, :], 0, st_full)
        
        for var in sparse_target_vars:
            vidx = sparse_target_vars.index(var)
            var_data = st_full_processed[:, vidx]  # (T, 32, 32)
            if split == "train":
                sparse_target_train_dict[var].append(var_data[None, ...])
            elif split == "valid":
                sparse_target_valid_dict[var].append(var_data[None, ...])
            else:
                sparse_target_test_dict[var].append(var_data[None, ...])
        
        # Dense target 처리
        dt_full = arr_dense_target[target_start:target_end]
        dt_full = dt_full[None, ...]
        if split == "train":
            dense_target_train.append(dt_full)
        elif split == "valid":
            dense_target_valid.append(dt_full)
        else:
            dense_target_test.append(dt_full)
        
        # High target 처리
        ht_full = arr_high_target[target_start:target_end]
        for var in high_target_vars:
            vidx = high_target_vars.index(var)
            var_data = ht_full[:, vidx]
            if split == "train":
                high_target_train_dict[var].append(var_data[None, ...])
            elif split == "valid":
                high_target_valid_dict[var].append(var_data[None, ...])
            else:
                high_target_test_dict[var].append(var_data[None, ...])
    
    print("\n[3] 저장을 시작합니다.\n")
    save_input_list(SAVE_DIR_TRAIN, "input_sparse", sparse_input_train, sparse_input_vars)
    save_input_list(SAVE_DIR_TRAIN, "input_stale",  stale_state_train, [SPARSE_STALE_VAR])
    save_input_list(SAVE_DIR_TRAIN, "input_dense",  dense_input_train, dense_input_vars)
    save_input_list(SAVE_DIR_TRAIN, "input_low",    low_input_train,   low_input_vars)
    
    save_target_dict(SAVE_DIR_TRAIN, "sparse_target", sparse_target_train_dict, sparse_target_vars)
    save_dense_target_as_one_file(SAVE_DIR_TRAIN, dense_target_train, "dense_target")
    save_target_dict(SAVE_DIR_TRAIN, "high_target", high_target_train_dict, high_target_vars)
    
    save_input_list(SAVE_DIR_VALID, "input_sparse", sparse_input_valid, sparse_input_vars)
    save_input_list(SAVE_DIR_VALID, "input_stale",  stale_state_valid, [SPARSE_STALE_VAR])
    save_input_list(SAVE_DIR_VALID, "input_dense",  dense_input_valid, dense_input_vars)
    save_input_list(SAVE_DIR_VALID, "input_low",    low_input_valid,   low_input_vars)
    
    save_target_dict(SAVE_DIR_VALID, "sparse_target", sparse_target_valid_dict, sparse_target_vars)
    save_dense_target_as_one_file(SAVE_DIR_VALID, dense_target_valid, "dense_target")
    save_target_dict(SAVE_DIR_VALID, "high_target", high_target_valid_dict, high_target_vars)
    
    save_input_list(SAVE_DIR_TEST, "input_sparse", sparse_input_test, sparse_input_vars)
    save_input_list(SAVE_DIR_TEST, "input_stale",  stale_state_test, [SPARSE_STALE_VAR])
    save_input_list(SAVE_DIR_TEST, "input_dense",  dense_input_test, dense_input_vars)
    save_input_list(SAVE_DIR_TEST, "input_low",    low_input_test,   low_input_vars)
    
    save_target_dict(SAVE_DIR_TEST, "sparse_target", sparse_target_test_dict, sparse_target_vars)
    save_dense_target_as_one_file(SAVE_DIR_TEST, dense_target_test, "dense_target")
    save_target_dict(SAVE_DIR_TEST, "high_target", high_target_test_dict, high_target_vars)
    
    print("\n[완료] 모든 Numpy 저장이 끝났습니다.")
    print(f"최종 Sparse Target Dataset은 '{SAVE_SPARSE_TARGET_ZERO}'로 저장될 예정입니다.")

if __name__ == "__main__":
    main()

# %%
