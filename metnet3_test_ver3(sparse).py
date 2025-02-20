import os
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================
# 예시: seed 고정
# ============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------------------
# CRPS, CSI 계산 함수 (동일)
# ----------------------------
def compute_crps(pred_probs: torch.Tensor, target_labels: torch.Tensor) -> float:
    B, C, H, W = pred_probs.shape
    assert target_labels.shape == (B, H, W)
    cdf_pred = torch.cumsum(pred_probs, dim=1)
    device = target_labels.device
    cdf_true = (torch.arange(C, device=device).view(1, C, 1, 1) >= target_labels.unsqueeze(1)).float()
    diff_sq = (cdf_pred - cdf_true) ** 2
    crps_val = diff_sq.mean().item()
    return crps_val

def compute_csi(pred_probs: torch.Tensor, target_labels: torch.Tensor,
                threshold_bin: int = 8,
                prob_threshold: float = 0.5) -> float:
    B, C, H, W = pred_probs.shape
    p_rain = pred_probs[:, threshold_bin:, :, :].sum(dim=1)  # (B,H,W)
    pred_positive = (p_rain >= prob_threshold)
    real_positive = (target_labels >= threshold_bin)
    tp = (pred_positive & real_positive).sum().item()
    fp = (pred_positive & (~real_positive)).sum().item()
    fn = ((~pred_positive) & real_positive).sum().item()
    denom = tp + fp + fn
    csi = tp / denom if denom > 0 else 0.0
    return csi

# ============================================================
# WeatherBenchDataset
#  - "보간 위치" + "25% 무작위 마스킹" 반영 (경로 수정)
# ============================================================
class WeatherBenchDataset(Dataset):
    def __init__(
        self,
        root_dir,
        # 보간된 위치 파일은 root_dir 내부가 아니라 Weather 경로 하위에 별도 존재
        sparse_input_indices_path="/projects/aiid/KIPOT_SKT/Weather/sparse_data_input/selected_indices_80.npy",
        sparse_target_indices_path="/projects/aiid/KIPOT_SKT/Weather/sparse_data_target/selected_indices_80_target.npy",
        is_training=True,
        random_seed=42,
    ):
        """
        root_dir:  trainset, validationset, testset 등 각각의 경로
        sparse_input_indices_path: 보간된 위치 인덱스 파일(156x156 해상도)
        sparse_target_indices_path: 보간된 위치 인덱스 파일(32x32 해상도)
        """
        self.root_dir = root_dir
        self.is_training = is_training
        np.random.seed(random_seed)  # 동일 마스킹 위해 seed 고정

        # ==========================
        # (1) 입력 (Input)
        # ==========================
        self.input_sparse = np.load(os.path.join(root_dir, 'input_sparse_normalized.npy'))  # (N,30,156,156)
        self.input_stale  = np.load(os.path.join(root_dir, 'input_stale_normalized.npy'))   # (N, 6,156,156)
        self.input_dense  = np.load(os.path.join(root_dir, 'input_dense_normalized.npy'))   # (N,36,156,156)
        self.input_low    = np.load(os.path.join(root_dir, 'input_low_normalized.npy'))     # (N,12,156,156)
        self.num_samples = self.input_sparse.shape[0]

        # ==========================
        # (2) 타겟 (Target)
        # ==========================
        self.t2m = np.load(os.path.join(root_dir, 'sparse_target', '2m_temperature.npy'))           # (N,6,32,32)
        self.d2m = np.load(os.path.join(root_dir, 'sparse_target', '2m_dewpoint_temperature.npy'))  # (N,6,32,32)
        self.precip32 = np.load(os.path.join(root_dir, 'sparse_target', 'total_precipitation.npy')) # (N,6,32,32)

        self.dense_target_36ch = np.load(os.path.join(root_dir, 'dense_target.npy'))  # (N,36,32,32)
        self.high_precip = np.load(os.path.join(root_dir, 'high_target', 'total_precipitation.npy'))# (N,6,128,128)
        self.hrrr_36ch   = self.dense_target_36ch  # 예시상 동일하게 사용

        # ==========================
        # (3) "보간 위치" 인덱스 로드 (절대 경로)
        # ==========================
        interp_input_indices = np.load(sparse_input_indices_path)   # (예: shape=(80,2))
        interp_target_indices = np.load(sparse_target_indices_path) # (예: shape=(80,2))

        # (A) input_mask 구성 (True=사용, False=제외)
        #     156x156 전체 True 후, 보간 위치는 False
        self.input_mask = np.ones((156,156), dtype=bool)
        for (r, c) in interp_input_indices:
            self.input_mask[r, c] = False  # 보간된 위치는 제외

        # (B) 25% 무작위 마스킹 (보간 아닌 실제 관측 위치 중 25% 더 제거)
        real_positions = np.argwhere(self.input_mask)  # (N_real,2)
        num_real = len(real_positions)
        num_drop = int(0.25 * num_real)
        drop_indices = np.random.choice(num_real, size=num_drop, replace=False)
        for di in drop_indices:
            rr, cc = real_positions[di]
            self.input_mask[rr, cc] = False

        # (C) target_mask (32x32)
        self.target_mask = np.ones((32,32), dtype=bool)
        for (r, c) in interp_target_indices:
            self.target_mask[r, c] = False

        # (D) 이름들 (예시)
        self.surface_bin_names = ('temperature_2m', 'dewpoint_2m')
        self.precipitation_bin_names = ('total_precipitation',)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # --------- (1) 입력 변환 + 마스킹 ---------
        in_sparse_np = self.input_sparse[idx].copy()  # shape (30,156,156)
        # 보간 + 25% 제거된 위치 => 0으로 세팅
        in_sparse_np[:, ~self.input_mask] = 0
        in_sparse = torch.from_numpy(in_sparse_np).float()

        in_stale = torch.from_numpy(self.input_stale[idx]).float()    # (6,156,156)
        in_dense = torch.from_numpy(self.input_dense[idx]).float()    # (36,156,156)
        in_low   = torch.from_numpy(self.input_low[idx]).float()      # (12,156,156)

        # --------- (2) 타깃 변환 + 마스킹 ---------
        #  (a) surface(2m, dewpoint, precip32) => shape (6,32,32)
        t2m_6h_np    = self.t2m[idx].copy()
        d2m_6h_np    = self.d2m[idx].copy()
        precip_6h_np = self.precip32[idx].copy()

        # 마지막 time step 마스킹 => -1
        t2m_6h_np[-1][~self.target_mask] = -1
        d2m_6h_np[-1][~self.target_mask] = -1
        precip_6h_np[-1][~self.target_mask] = -1

        t2m_6h    = torch.from_numpy(t2m_6h_np).long()
        d2m_6h    = torch.from_numpy(d2m_6h_np).long()
        precip_6h = torch.from_numpy(precip_6h_np).long()

        #  (b) dense_target_36ch => (36,32,32), 예시상 분류라면 -1 마스킹
        dense_36ch_np = self.dense_target_36ch[idx].copy()
        for ch in range(dense_36ch_np.shape[0]):
            dense_36ch_np[ch][~self.target_mask] = -1
        dense_target_36ch = torch.from_numpy(dense_36ch_np).long()

        #  (c) high_precip => (6,128,128)는 별도 해상도 => 그대로 사용
        high_precip_6h = torch.from_numpy(self.high_precip[idx]).long()

        #  (d) hrrr_36ch => (36,32,32), float. 여기서는 -1 마스킹 안함(옵션)
        hrrr_36ch = torch.from_numpy(self.hrrr_36ch[idx]).float()

        # --------- (3) 임의 lead time ---------
        lead_time = torch.tensor(np.random.randint(0, 722), dtype=torch.long)

        sample = {
            "lead_time": lead_time,
            "input_sparse": in_sparse,
            "input_stale":  in_stale,
            "input_dense":  in_dense,
            "input_low":    in_low,
            "precipitation_targets": {
                "total_precipitation": high_precip_6h[-1],  # (128,128)
            },
            "surface_targets": {
                "temperature_2m": t2m_6h[-1],  # (32,32)
                "dewpoint_2m":    d2m_6h[-1],  # (32,32)
            },
            "hrrr_target": hrrr_36ch,   # (36,32,32)
        }
        return sample


# ============================================================
# (4) DataLoader 생성
# ============================================================
train_root = r"/projects/aiid/KIPOT_SKT/Weather/trainset"
val_root   = r"/projects/aiid/KIPOT_SKT/Weather/validationset"
test_root  = r"/projects/aiid/KIPOT_SKT/Weather/testset"

train_dataset = WeatherBenchDataset(
    root_dir=train_root,
    sparse_input_indices_path="/projects/aiid/KIPOT_SKT/Weather/sparse_data_input/selected_indices_80.npy",
    sparse_target_indices_path="/projects/aiid/KIPOT_SKT/Weather/sparse_data_target/selected_indices_80_target.npy",
    is_training=True,
    random_seed=SEED
)

val_dataset = WeatherBenchDataset(
    root_dir=val_root,
    sparse_input_indices_path="/projects/aiid/KIPOT_SKT/Weather/sparse_data_input/selected_indices_80.npy",
    sparse_target_indices_path="/projects/aiid/KIPOT_SKT/Weather/sparse_data_target/selected_indices_80_target.npy",
    is_training=False,
    random_seed=SEED
)

test_dataset = WeatherBenchDataset(
    root_dir=test_root,
    sparse_input_indices_path="/projects/aiid/KIPOT_SKT/Weather/sparse_data_input/selected_indices_80.npy",
    sparse_target_indices_path="/projects/aiid/KIPOT_SKT/Weather/sparse_data_target/selected_indices_80_target.npy",
    is_training=False,
    random_seed=SEED
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=3)
val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False, num_workers=3)
test_loader  = DataLoader(test_dataset,  batch_size=2, shuffle=False, num_workers=3)

# 이어서 MetNet3 모델 정의, 학습 루프 등 기존 코드 재사용
# ...


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,  num_workers=3)
val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False, num_workers=3)
test_loader  = DataLoader(test_dataset,  batch_size=2, shuffle=False, num_workers=3)

from metnet3_original import MetNet3

# 이미 위에서 seed 설정했으므로, 여기서는 추가 설정만 예시
# import random
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 생성
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
    # 채널 설정
    hrrr_channels = 36,
    input_2496_channels = 30,
    input_4996_channels = 12,
    surface_and_hrrr_target_spatial_size = 32,
    precipitation_target_bins = dict(
        total_precipitation = 512,  # 512 bins
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
).to(device)

optimizer = torch.optim.Adam(metnet3.parameters(), lr=1e-4)

best_crps = float('inf')
best_csi  = 0.0
best_model_state = None

min_epochs = 30
max_no_improve = 5
epoch = 0
no_improve_count = 0

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
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"[Epoch {epoch}] Train Loss = {avg_epoch_loss:.4f}, Breakdown = {loss_breakdown}")

    # Validation
    metnet3.eval()
    total_crps = 0.0
    total_csi = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            lead_times = batch['lead_time'].to(device)
            in_sparse  = batch['input_sparse'].to(device)
            in_stale   = batch['input_stale'].to(device)
            in_dense   = batch['input_dense'].to(device)
            in_low     = batch['input_low'].to(device)

            precip_targets = { k: v.to(device) for k,v in batch['precipitation_targets'].items() }

            # 예측 (타겟 없이)
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
            tgt = precip_targets['total_precipitation']          # (B, H, W)

            # CRPS
            crps_batch = compute_crps(probs, tgt)
            # CSI
            csi_batch = compute_csi(probs, tgt, threshold_bin=8, prob_threshold=0.5)

            total_crps += crps_batch
            total_csi += csi_batch
            num_batches += 1

    mean_crps = total_crps / num_batches
    mean_csi  = total_csi / num_batches

    print(f"[Epoch {epoch}] Validation CRPS: {mean_crps:.4f}, CSI: {mean_csi:.4f}")

    # 개선 여부 확인
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
        print(f"No improvement: {no_improve_count} consecutive epoch(s).")

    if epoch >= min_epochs and no_improve_count >= max_no_improve:
        print(f"Early stopping after {epoch} epochs.")
        break

# ============================================================
# (마지막) 최종 모델 저장
# ============================================================
if best_model_state is not None:
    metnet3.load_state_dict(best_model_state)

save_dir = "/projects/aiid/KIPOT_SKT/Weather/test_outputs"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "metnet3_final.pth")
torch.save(metnet3.state_dict(), save_path)
print(f"Final model saved to {save_path}")