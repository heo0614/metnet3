import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from netCDF4 import Dataset as ncDataset

# kriging 보간 관련 함수 (추후 실제 함수로 대체)
from kriging.kriging_interpolation import dataset_to_array_kriging, KrigingConfig

# ============================================================
# (1) CRPS & CSI 계산 함수 (동일)
# ============================================================
def compute_crps(pred_probs: torch.Tensor, target_labels: torch.Tensor) -> float:
    B, C, H, W = pred_probs.shape
    assert target_labels.shape == (B, H, W)
    cdf_pred = torch.cumsum(pred_probs, dim=1)
    device = target_labels.device
    cdf_true = (torch.arange(C, device=device).view(1, C, 1, 1) >= target_labels.unsqueeze(1)).float()
    diff_sq = (cdf_pred - cdf_true) ** 2
    return diff_sq.mean().item()

def compute_csi(pred_probs: torch.Tensor, target_labels: torch.Tensor,
                threshold_bin: int = 8,
                prob_threshold: float = 0.5) -> float:
    B, C, H, W = pred_probs.shape
    p_rain = pred_probs[:, threshold_bin:, :, :].sum(dim=1)
    pred_positive = (p_rain >= prob_threshold)
    real_positive = (target_labels >= threshold_bin)
    tp = (pred_positive & real_positive).sum().item()
    fp = (pred_positive & (~real_positive)).sum().item()
    fn = ((~pred_positive) & real_positive).sum().item()
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0

# ============================================================
# (2) Dataset 정의: Preprocessed NC 파일 기반, 슬라이딩 윈도우 6시간 샘플
# ============================================================
class PreprocessedWeatherBenchDataset(Dataset):
    def __init__(self, root_dir, indices=None, window_size=6, config_kriging=None):
        """
        root_dir: preprocessed 파일들이 있는 폴더 (예: /projects/aiid/KIPOT_SKT/Weather)
        indices: 슬라이딩 윈도우의 시작 인덱스 목록. 지정하지 않으면 전체 [0, T-window_size+1) 사용.
        window_size: 한 샘플에 사용할 연속 시간 수 (여기서는 6)
        config_kriging: kriging 보간에 사용할 설정 (KrigingConfig 객체)
        """
        self.root_dir = root_dir
        self.window_size = window_size
        self.config_kriging = config_kriging

        # 파일 경로 설정
        self.path_sparse  = os.path.join(root_dir, 'preprocessed', 'sparse_input_preprocessed.nc')
        self.path_dense   = os.path.join(root_dir, 'preprocessed', 'dense_input_preprocessed.nc')
        self.path_low     = os.path.join(root_dir, 'preprocessed', 'low_input_preprocessed.nc')
        self.path_high    = os.path.join(root_dir, 'preprocessed', 'high_target_preprocessed.nc')
        self.path_sparse_tgt = os.path.join(root_dir, 'preprocessed', 'sparse_target_preprocessed.nc')
        self.path_dense_tgt  = os.path.join(root_dir, 'preprocessed', 'dense_target_preprocessed.nc')

        # NC파일 오픈 (전체 데이터를 메모리에 올림)
        nc_sparse  = ncDataset(self.path_sparse, mode='r')
        nc_dense   = ncDataset(self.path_dense, mode='r')
        nc_low     = ncDataset(self.path_low, mode='r')
        nc_high    = ncDataset(self.path_high, mode='r')
        nc_sparse_tgt = ncDataset(self.path_sparse_tgt, mode='r')
        nc_dense_tgt  = ncDataset(self.path_dense_tgt, mode='r')

        # --- Sparse Input ---
        # 변수 순서: '2m_temperature', 'surface_pressure', 'total_precipitation',
        #           'u_component_of_wind', 'v_component_of_wind', '2m_dewpoint_temperature'
        varnames_sparse = ['2m_temperature', 'surface_pressure', 'total_precipitation',
                           'u_component_of_wind', 'v_component_of_wind', '2m_dewpoint_temperature']
        sparse_list = [nc_sparse.variables[v][:] for v in varnames_sparse]
        self.sparse_data = np.stack(sparse_list, axis=1)  # shape=(T, C, 156, 156)

        # --- Dense Input ---
        varnames_dense = ['geopotential', 'land_sea_mask', 'temperature',
                          '10m_u_component_of_wind', '10m_v_component_of_wind', 'specific_humidity']
        dense_list = [nc_dense.variables[v][:] for v in varnames_dense]
        self.dense_data = np.stack(dense_list, axis=1)  # shape=(T, C, 156, 156)

        # --- Low Input ---
        varnames_low = ['total_cloud_cover', 'total_precipitation']
        low_list = [nc_low.variables[v][:] for v in varnames_low]
        self.low_data = np.stack(low_list, axis=1)  # shape=(T, C, 156, 156)

        # --- High Target ---
        self.high_data = nc_high.variables['total_precipitation'][:]  # shape=(T, 128, 128)

        # --- Sparse Target ---
        varnames_sparse_tgt = ['2m_temperature', 'total_precipitation', '2m_dewpoint_temperature']
        sparse_tgt_list = [nc_sparse_tgt.variables[v][:] for v in varnames_sparse_tgt]
        self.sparse_tgt_data = np.stack(sparse_tgt_list, axis=1)  # shape=(T, C, 32, 32)

        # --- Dense Target ---
        varnames_dense_tgt = ['geopotential', 'land_sea_mask', 'temperature',
                              '10m_u_component_of_wind', '10m_v_component_of_wind', 'specific_humidity']
        dense_tgt_list = [nc_dense_tgt.variables[v][:] for v in varnames_dense_tgt]
        self.dense_tgt_data = np.stack(dense_tgt_list, axis=1)  # shape=(T, C, 32, 32)

        # 전체 시간 T
        self.T = self.sparse_data.shape[0]
        # 슬라이딩 윈도우 샘플 수: T - window_size + 1
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(self.T - window_size + 1)
        self.num_samples = len(self.indices)

        # 각 샘플에 대해, 입력 sparse의 첫 시간(윈도우의 시작 시간)의 '2m_temperature'(채널 0)에서
        # 값이 0이 아닌 위치 중 80%를 무작위 선택하여 valid 위치로 저장
        self.valid_idx = {}
        for i, t in enumerate(self.indices):
            sample_channel = self.sparse_data[t, 0]  # shape=(156,156)
            nonzero_idx = np.array(np.nonzero(sample_channel)).T
            num_select = int(0.8 * nonzero_idx.shape[0])
            if num_select > 0:
                selected = nonzero_idx[np.random.choice(nonzero_idx.shape[0], num_select, replace=False)]
            else:
                selected = nonzero_idx
            self.valid_idx[t] = selected

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 슬라이딩 윈도우 시작 인덱스
        t0 = self.indices[idx]
        t1 = t0 + self.window_size  # t0 ~ t0+window_size-1 (6시간)

        # 각 변수에서 6시간 블록 슬라이스 (numpy 배열)
        # 먼저 kriging 보간 적용 (kriging은 numpy 단계에서 수행)
        # 예: in_sparse_block = dataset_to_array_kriging( 원본 6시간 블록, config )
        in_sparse_block = self.sparse_data[t0:t1]  # (6, C, 156, 156)
        if self.config_kriging is not None:
            in_sparse_block = dataset_to_array_kriging(in_sparse_block, self.config_kriging)
        in_dense_block = self.dense_data[t0:t1]      # (6, C, 156, 156)
        in_low_block = self.low_data[t0:t1]          # (6, C, 156, 156)

        # 타겟: 여기서는 sparse target와 high target, dense target 모두 6시간 블록으로 구성
        sparse_tgt_block = self.sparse_tgt_data[t0:t1]  # (6, C, 32, 32)
        dense_tgt_block = self.dense_tgt_data[t0:t1]      # (6, C, 32, 32)
        high_tgt_block = self.high_data[t0:t1]            # (6, 128, 128)

        # 모델 입력: flatten 시간 축 into channel dimension.
        # 예를 들어, in_sparse_block 원래 shape (6, C, 156,156) -> (6*C, 156,156)
        in_sparse_flat = in_sparse_block.reshape(-1, in_sparse_block.shape[2], in_sparse_block.shape[3])
        in_dense_flat = in_dense_block.reshape(-1, in_dense_block.shape[2], in_dense_block.shape[3])
        in_low_flat = in_low_block.reshape(-1, in_low_block.shape[2], in_low_block.shape[3])
        # input stale: sparse input의 '2m_temperature' 채널 from 첫 시간, 그대로 사용 (156,156)
        in_stale = torch.from_numpy(self.sparse_data[t0, 0]).float()
        
        # 타겟은 6시간 블록 그대로; 필요에 따라 flatten 할 수도 있음.
        # 여기서는 loss 계산 시에도 시간 축을 유지하지 않고 flatten (예: (6*target_channels, H, W))
        sparse_tgt_flat = sparse_tgt_block.reshape(-1, sparse_tgt_block.shape[2], sparse_tgt_block.shape[3])
        
        # 추가 25% 마스킹: valid 위치는 첫 시간의 valid_idx 사용
        valid_idx = self.valid_idx[t0]  # numpy array, shape=(num_valid, 2)
        
        # 이제 torch.Tensor 변환
        in_sparse_flat = torch.from_numpy(in_sparse_flat).float()   # (6*C, 156,156)
        in_dense_flat = torch.from_numpy(in_dense_flat).float()       # (6*C, 156,156)
        in_low_flat = torch.from_numpy(in_low_flat).float()           # (6*C, 156,156)
        sparse_tgt_flat = torch.from_numpy(sparse_tgt_flat).long()      # (6*C_tgt, 32,32)  (C_tgt는 sparse target 채널 수)
        # in_stale는 이미 torch.Tensor
        
        # Lead time: t0 (starting hour)
        lead_time = torch.tensor(t0, dtype=torch.long)
        
        sample = {
            "lead_time": lead_time,
            "input_sparse": in_sparse_flat,    # (6*C, 156,156)
            "input_stale": in_stale,           # (156,156)
            "input_dense": in_dense_flat,      # (6*C, 156,156)
            "input_low": in_low_flat,          # (6*C, 156,156)
            "sparse_target": sparse_tgt_flat,  # (6*C_tgt, 32,32)
            "dense_target": dense_tgt_block.reshape(-1, dense_tgt_block.shape[2], dense_tgt_block.shape[3]),  # (6*..., 32,32)
            "high_target": high_tgt_block.reshape(-1, high_tgt_block.shape[1], high_tgt_block.shape[2]),         # (6, 128,128) -> if needed flatten time too
            "valid_idx": valid_idx             # (num_valid, 2) from 첫 시간
        }
        return sample

# ============================================================
# (3) Train/Val/Test 분할 (50%/25%/25%)
# ============================================================
data_root = r"/projects/aiid/KIPOT_SKT/Weather"
total_samples = 1464
# 슬라이딩 윈도우 샘플 수 = total_samples - window_size + 1
window_size = 6
all_indices = np.arange(total_samples - window_size + 1)
train_end = int(0.5 * len(all_indices))
val_end = int(0.75 * len(all_indices))
train_indices = all_indices[:train_end]
val_indices = all_indices[train_end:val_end]
test_indices = all_indices[val_end:]

# kriging 설정 (예시)
kriging_config = KrigingConfig(
    variogram_model='exponential',
    sampling_ratio=0.025,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    exclude_vars=['land_sea_mask']
)

train_dataset = PreprocessedWeatherBenchDataset(data_root, indices=train_indices, window_size=window_size, config_kriging=kriging_config)
val_dataset = PreprocessedWeatherBenchDataset(data_root, indices=val_indices, window_size=window_size, config_kriging=kriging_config)
test_dataset = PreprocessedWeatherBenchDataset(data_root, indices=test_indices, window_size=window_size, config_kriging=kriging_config)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=3)
val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=3)
# ============================================================
# (4) MetNet3 모델 준비  
# (주의: 모델의 입력 채널 수는 flatten된 시간*원래 채널 수에 맞게 수정)
# ============================================================
from metnet3_original import MetNet3

# 예시: 만약 sparse_input 원래 채널 수가 6개라면,
# flatten 후 입력 채널 수는 6 * 6 = 36.
# dense_input 원래 채널 수 6 -> 6*6 = 36, low_input 원래 채널 수 2 -> 2*6 = 12.
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
    # 채널 수를 실제 데이터에 맞게 조정
    hrrr_channels = 36,           # dense: 6*6
    input_2496_channels = 36,     # sparse: 6*6
    input_4996_channels = 12,     # low: 2*6
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metnet3.to(device)
optimizer = torch.optim.Adam(metnet3.parameters(), lr=1e-4)

# ============================================================
# (5) Training Loop: 전체 시간 샘플에 대해 학습 (6시간씩 입력, 6시간 타겟)
# ============================================================
best_crps = float('inf')
best_csi = 0.0
best_model_state = None
min_epochs = 30
max_no_improve = 5
no_improve_count = 0

for epoch in range(1, 1000):
    metnet3.train()
    epoch_loss = 0.0
    for batch in train_loader:
        # ----- 1. 데이터 준비 -----
        # 각 배치의 샘플은 이미 6시간 블록에서 시간 축이 flatten된 상태
        lead_times = batch['lead_time'].to(device)  # (B,)
        in_sparse = batch['input_sparse'].clone().to(device)  # (B, 6*C, 156,156)
        in_dense = batch['input_dense'].to(device)            # (B, 6*C, 156,156)
        in_low = batch['input_low'].to(device)                # (B, 6*C, 156,156)
        in_stale = batch['input_stale'].to(device)            # (B, 156,156)
        sparse_target = batch['sparse_target'].to(device)     # (B, 6*C_tgt, 32,32)
        
        # kriging 보간은 이미 dataset __getitem__에서 적용했으므로 여기선 생략
        
        # 각 샘플별로 미리 저장된 valid_idx (첫 시간 기준)에서 추가 25% 선택하여 마스킹
        mask_list = []
        B_size, _, H, W = in_sparse.shape
        for i in range(B_size):
            valid_idx = batch['valid_idx'][i]  # numpy array, shape=(num_valid, 2)
            num_valid = valid_idx.shape[0]
            num_mask = int(0.25 * num_valid)
            if num_mask > 0:
                chosen = valid_idx[np.random.choice(num_valid, num_mask, replace=False)]
            else:
                chosen = valid_idx
            mask_list.append(chosen)
            for r, c in chosen:
                in_sparse[i, :, r, c] = 0.0

        # ----- 2. Forward & Loss 계산 (마스킹된 위치만 loss 계산) -----
        optimizer.zero_grad()
        output = metnet3(
            lead_times = lead_times,
            hrrr_input_2496 = in_dense,
            hrrr_stale_state = in_stale,
            input_2496 = in_sparse,
            input_4996 = in_low,
        )
        logits = output.precipitation['total_precipitation']  # (B, C_out, H_out, W_out)
        probs = F.softmax(logits, dim=1)
        
        loss_total = 0.0
        for i in range(B_size):
            # 생성한 mask (H_out, W_out); assume H_out, W_out match those of target
            mask = torch.zeros((W, W), dtype=torch.bool, device=device)
            for r, c in mask_list[i]:
                mask[r, c] = True
            logits_i = logits[i].permute(1, 2, 0).reshape(-1, logits.shape[1])
            target_i = sparse_target[i].reshape(-1)
            mask_flat = mask.reshape(-1)
            logits_masked = logits_i[mask_flat]
            target_masked = target_i[mask_flat]
            if logits_masked.shape[0] > 0:
                loss_i = F.cross_entropy(logits_masked, target_masked)
                loss_total += loss_i
        loss_total.backward()
        optimizer.step()
        epoch_loss += loss_total.item()
    
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"[Epoch {epoch}] Train Loss = {avg_epoch_loss:.4f}")
    
    # ----- (A) Validation: 전체 validation 셋에 대해 CRPS & CSI 계산 -----
    metnet3.eval()
    total_crps = 0.0
    total_csi = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            lead_times_val = batch['lead_time'].to(device)
            in_sparse_val = batch['input_sparse'].to(device)
            in_stale_val = batch['input_stale'].to(device)
            in_dense_val = batch['input_dense'].to(device)
            in_low_val = batch['input_low'].to(device)
            precip_targets_val = {'total_precipitation': batch['sparse_target'].to(device)}
            pred = metnet3(
                lead_times = lead_times_val,
                hrrr_input_2496 = in_dense_val,
                hrrr_stale_state = in_stale_val,
                input_2496 = in_sparse_val,
                input_4996 = in_low_val,
            )
            logits_val = pred.precipitation['total_precipitation']
            probs_val = F.softmax(logits_val, dim=1)
            tgt_val = precip_targets_val['total_precipitation']
            crps_batch = compute_crps(probs_val, tgt_val)
            csi_batch = compute_csi(probs_val, tgt_val, threshold_bin=8, prob_threshold=0.5)
            total_crps += crps_batch
            total_csi += csi_batch
            num_batches += 1
    mean_crps = total_crps / num_batches
    mean_csi = total_csi / num_batches
    print(f"[Epoch {epoch}] Validation CRPS: {mean_crps:.4f}, CSI: {mean_csi:.4f}")
    
    # ----- (B) 개선 여부 체크 및 Early Stopping -----
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
        print(f"Improved => CRPS={best_crps:.4f}, CSI={best_csi:.4f}")
    else:
        no_improve_count += 1
        print(f"No improvement for {no_improve_count} epoch(s).")
    if epoch >= min_epochs and no_improve_count >= max_no_improve:
        print(f"Early stopping at epoch {epoch}.")
        break

# ============================================================
# (6) 최종 모델 저장
# ============================================================
if best_model_state is not None:
    metnet3.load_state_dict(best_model_state)
save_dir = "/projects/aiid/KIPOT_SKT/Weather/test_outputs"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "metnet3_final.pth")
torch.save(metnet3.state_dict(), save_path)
print(f"Final model saved to {save_path}")
