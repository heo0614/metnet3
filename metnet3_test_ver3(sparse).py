import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from kriging.kriging_interpolation import dataset_to_array_kriging, KrigingConfig
from kriging import functions, kriging_interpolation

# ============================================================
# (1) CRPS 계산 함수 (Discrete)
# ============================================================
def compute_crps(pred_probs: torch.Tensor, target_labels: torch.Tensor) -> float:
    B, C, H, W = pred_probs.shape
    assert target_labels.shape == (B, H, W), "target_labels shape mismatch."

    cdf_pred = torch.cumsum(pred_probs, dim=1)

    device = target_labels.device
    cdf_true = (torch.arange(C, device=device).view(1, C, 1, 1) >= target_labels.unsqueeze(1)).float()

    diff_sq = (cdf_pred - cdf_true) ** 2
    crps_val = diff_sq.mean().item()
    return crps_val

# ============================================================
# (2) CSI 계산 함수
# ============================================================
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
# (3) Dataset 정의
# ============================================================
class WeatherBenchDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # ----- (A) 입력 (Input) -----
        self.input_sparse = np.load(os.path.join(root_dir, 'input_sparse_normalized.npy'), mmap_mode='r')  # (N,30,156,156)
        self.input_stale  = np.load(os.path.join(root_dir, 'input_stale_normalized.npy'),  mmap_mode='r')  # (N, 6,156,156)
        self.input_dense  = np.load(os.path.join(root_dir, 'input_dense_normalized.npy'),  mmap_mode='r')  # (N,36,156,156)
        self.input_low    = np.load(os.path.join(root_dir, 'input_low_normalized.npy'),    mmap_mode='r')  # (N,12,156,156)
        self.num_samples  = self.input_sparse.shape[0]

        # ----- (B) 타겟 (Target) -----
        self.t2m         = np.load(os.path.join(root_dir, 'sparse_target', '2m_temperature.npy'),          mmap_mode='r')  # (N,6,32,32)
        self.d2m         = np.load(os.path.join(root_dir, 'sparse_target', '2m_dewpoint_temperature.npy'), mmap_mode='r')  # (N,6,32,32)
        self.precip32    = np.load(os.path.join(root_dir, 'sparse_target', 'total_precipitation.npy'),     mmap_mode='r')  # (N,6,32,32)
        self.dense_tgt   = np.load(os.path.join(root_dir, 'dense_target.npy'),                             mmap_mode='r')  # (N,36,32,32)
        self.high_precip = np.load(os.path.join(root_dir, 'high_target', 'total_precipitation.npy'),       mmap_mode='r')  # (N,6,128,128)

        self.hrrr_36ch   = self.dense_tgt  # 예시: dense_tgt를 hrrr로 사용

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # ---- (1) 입력 텐서 ----
        in_sparse = torch.from_numpy(self.input_sparse[idx]).float()  # (30,156,156)
        in_stale  = torch.from_numpy(self.input_stale[idx]).float()   # ( 6,156,156)
        in_dense  = torch.from_numpy(self.input_dense[idx]).float()   # (36,156,156)
        in_low    = torch.from_numpy(self.input_low[idx]).float()     # (12,156,156)

        # ---- (2) 타겟 텐서 ----
        t2m_6h    = torch.from_numpy(self.t2m[idx]).long()      # (6,32,32)
        d2m_6h    = torch.from_numpy(self.d2m[idx]).long()      # (6,32,32)
        precip_6h = torch.from_numpy(self.precip32[idx]).long() # (6,32,32)

        dense_target_36ch = torch.from_numpy(self.dense_tgt[idx]).long()  # (36,32,32)
        high_precip_6h    = torch.from_numpy(self.high_precip[idx]).long()# (6,128,128)
        hrrr_36ch         = torch.from_numpy(self.hrrr_36ch[idx]).float() # (36,32,32)

        # 임의 리드타임(0~721)
        lead_time = torch.tensor(np.random.randint(0, 722), dtype=torch.long)

        sample = {
            "lead_time": lead_time,
            "input_sparse": in_sparse,  # (30,156,156)
            "input_stale":  in_stale,   # ( 6,156,156)
            "input_dense":  in_dense,   # (36,156,156)
            "input_low":    in_low,     # (12,156,156),
            "precipitation_targets": {
                "total_precipitation": high_precip_6h[-1]  # (128,128) 마지막 스텝
            },
            "surface_targets": {
                "temperature_2m": t2m_6h[-1],  # (32,32)
                "dewpoint_2m":    d2m_6h[-1],  # (32,32)
            },
            "hrrr_target": hrrr_36ch,   # (36,32,32)
        }
        return sample


# ============================================================
# (4) 훈련/검증/테스트용 Dataset
# ============================================================
train_root = r"/projects/aiid/KIPOT_SKT/Weather/trainset"
val_root   = r"/projects/aiid/KIPOT_SKT/Weather/validationset"
test_root  = r"/projects/aiid/KIPOT_SKT/Weather/testset"

train_dataset = WeatherBenchDataset(train_root)
val_dataset   = WeatherBenchDataset(val_root)
test_dataset  = WeatherBenchDataset(test_root)

print(f"train samples: {len(train_dataset)}, val samples: {len(val_dataset)}, test samples: {len(test_dataset)}")


# ============================================================
# (5) MetNet3 모델 준비
#    - 기존 예시의 metnet3_original.py import
# ============================================================
from metnet3_original import MetNet3

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
# (6) Training Loop (샘플 단위) + Early Stopping
# ============================================================
best_crps = float('inf')
best_csi  = 0.0
best_model_state = None

min_epochs = 30
max_no_improve = 5
epoch = 0
no_improve_count = 0

config = KrigingConfig(
    variogram_model='exponential',
    sampling_ratio=0.025,
    # ...
)


while True:
    epoch += 1
    metnet3.train()

    # ---------------------------
    # [추가] 매 epoch마다 156x156 중 25% 마스킹
    # ---------------------------
    # shape=(156,156), True=마스킹할 위치
    random_mask_2d = (np.random.rand(156,156) < 0.25)

    train_indices = np.arange(len(train_dataset))
    np.random.shuffle(train_indices)

    total_loss = 0.0
    for i in train_indices:
        sample = train_dataset[i]

        # sparse input
        in_sparse = sample["input_sparse"].clone()  # (30,156,156) 텐서
        # 채널 차원(30)에 대해 random_mask_2d 적용\
        in_sparse = dataset_to_array_kriging(in_sparse, config)

        in_sparse[:, random_mask_2d] = 0.0

        # 나머지
        in_stale  = sample["input_stale"]
        in_dense  = sample["input_dense"]
        in_low    = sample["input_low"]

        lead_times = sample["lead_time"]
        precip_targets  = sample["precipitation_targets"]
        surface_targets = sample["surface_targets"]
        hrrr_target = sample["hrrr_target"]

        # (배치 차원 추가)
        lead_times = lead_times.unsqueeze(0).to(device)
        in_sparse  = in_sparse.unsqueeze(0).to(device)
        in_stale   = in_stale.unsqueeze(0).to(device)
        in_dense   = in_dense.unsqueeze(0).to(device)
        in_low     = in_low.unsqueeze(0).to(device)

        precip_targets  = { k: v.unsqueeze(0).to(device) for k,v in precip_targets.items() }
        surface_targets = { k: v.unsqueeze(0).to(device) for k,v in surface_targets.items() }
        hrrr_target     = hrrr_target.unsqueeze(0).to(device)

        optimizer.zero_grad()
        loss, loss_breakdown = metnet3(
            lead_times            = lead_times,
            hrrr_input_2496       = in_dense,
            hrrr_stale_state      = in_stale,
            input_2496            = in_sparse,
            input_4996            = in_low,
            precipitation_targets = precip_targets,
            surface_targets       = surface_targets,
            hrrr_target           = hrrr_target,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataset)
    print(f"[Epoch {epoch}] Train Loss = {avg_train_loss:.4f}")

    # ----- (B) Validation: CRPS & CSI -----
    metnet3.eval()
    total_crps = 0.0
    total_csi  = 0.0
    num_val_samples = len(val_dataset)

    with torch.no_grad():
        for i in range(num_val_samples):
            sample = val_dataset[i]
            # val에서는 마스킹 X
            lead_times = sample["lead_time"].unsqueeze(0).to(device)
            in_sparse  = sample["input_sparse"].unsqueeze(0).to(device)
            in_stale   = sample["input_stale"].unsqueeze(0).to(device)
            in_dense   = sample["input_dense"].unsqueeze(0).to(device)
            in_low     = sample["input_low"].unsqueeze(0).to(device)

            pred = metnet3(
                lead_times       = lead_times,
                hrrr_input_2496  = in_dense,
                hrrr_stale_state = in_stale,
                input_2496       = in_sparse,
                input_4996       = in_low,
            )

            precipitation_preds = pred.precipitation
            logits = precipitation_preds['total_precipitation']  # (1, C, H, W)
            probs = F.softmax(logits, dim=1)

            true_precip = sample["precipitation_targets"]["total_precipitation"].unsqueeze(0).to(device)

            crps_batch = compute_crps(probs, true_precip)
            total_crps += crps_batch

            csi_batch = compute_csi(probs, true_precip, threshold_bin=8, prob_threshold=0.5)
            total_csi += csi_batch

        mean_crps = total_crps / num_val_samples
        mean_csi  = total_csi  / num_val_samples

    print(f"[Epoch {epoch}] Validation CRPS: {mean_crps:.4f}, CSI: {mean_csi:.4f}")

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
# (7) 최종 모델 저장
# ============================================================
if best_model_state is not None:
    metnet3.load_state_dict(best_model_state)

save_dir = "/projects/aiid/KIPOT_SKT/Weather/test_outputs"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "metnet3_final.pth")
torch.save(metnet3.state_dict(), save_path)
print(f"Final model saved to {save_path}")
