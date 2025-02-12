import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# ============================================================
# (1) CRPS 계산 함수 (Discrete)
# ============================================================
def compute_crps(pred_probs: torch.Tensor, target_labels: torch.Tensor) -> float:
    """
    pred_probs: (B, C, H, W) - softmax를 취한 뒤의 예측 확률분포
    target_labels: (B, H, W) - 각 픽셀이 0~C-1 범위의 클래스 레이블
    반환: CRPS (스칼라, float)
    """
    # shape 체크
    B, C, H, W = pred_probs.shape
    assert target_labels.shape == (B, H, W)
    
    # (1) CDF(pred): (B, C, H, W)
    cdf_pred = torch.cumsum(pred_probs, dim=1)
    
    # (2) CDF(true): ground truth가 g 라면, cdf_true[k] = 1(g <= k)
    #    (k >= g 이면 1, 아니면 0)
    #    -> shape=(1, C, 1, 1) >= (B, 1, H, W) 브로드캐스팅
    #    -> 결과 shape (B, C, H, W)
    device = target_labels.device
    cdf_true = (torch.arange(C, device=device).view(1, C, 1, 1) >= target_labels.unsqueeze(1)).float()
    
    # (3) 차이 제곱
    diff_sq = (cdf_pred - cdf_true) ** 2   # (B, C, H, W)
    
    # (4) CRPS = diff_sq의 전체 평균
    crps_val = diff_sq.mean().item()  # 평균으로 계산
    return crps_val
# ============================================================
# (1) Dataset 정의 (사용자 코드 예시 그대로)
# ============================================================
class WeatherBenchDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

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
        self.t2m = np.load(os.path.join(root_dir, 'sparse_target', '2m_temperature.npy'))             # (N,6,32,32)
        self.d2m = np.load(os.path.join(root_dir, 'sparse_target', '2m_dewpoint_temperature.npy'))    # (N,6,32,32)
        self.precip32 = np.load(os.path.join(root_dir, 'sparse_target', 'total_precipitation.npy'))   # (N,6,32,32)

        self.dense_target_36ch = np.load(os.path.join(root_dir, 'dense_target.npy'))  # (N,36,32,32)
        self.high_precip = np.load(os.path.join(root_dir, 'high_target', 'total_precipitation.npy'))  # (N,6,128,128)
        self.hrrr_36ch   = self.dense_target_36ch  # 예시상 동일하게 사용

        self.surface_bin_names = ('temperature_2m', 'dewpoint_2m')
        self.precipitation_bin_names = ('total_precipitation',)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # ---- 입력 텐서 변환 ----
        in_sparse = torch.from_numpy(self.input_sparse[idx]).float()  # (30,156,156)
        in_stale  = torch.from_numpy(self.input_stale[idx]).float()   # ( 6,156,156)
        in_dense  = torch.from_numpy(self.input_dense[idx]).float()   # (36,156,156)
        in_low    = torch.from_numpy(self.input_low[idx]).float()     # (12,156,156)

        # ---- 타겟 텐서 변환 ---- (여기서는 이미 bin index라고 가정하여 long으로 변환)
        t2m_6h    = torch.from_numpy(self.t2m[idx]).long()      # (6,32,32)
        d2m_6h    = torch.from_numpy(self.d2m[idx]).long()      # (6,32,32)
        precip_6h = torch.from_numpy(self.precip32[idx]).long() # (6,32,32)

        dense_target_36ch = torch.from_numpy(self.dense_target_36ch[idx]).long()  # (36,32,32)
        high_precip_6h    = torch.from_numpy(self.high_precip[idx]).long()        # (6,128,128)
        hrrr_36ch         = torch.from_numpy(self.hrrr_36ch[idx]).float()         # (36,32,32)

        # ---- 임의 리드타임 예시 ----
        lead_time = torch.tensor(np.random.randint(0, 722), dtype=torch.long)

        sample = {
            "lead_time": lead_time,
            "input_sparse": in_sparse,  # (30,156,156)
            "input_stale":  in_stale,   # ( 6,156,156)
            "input_dense":  in_dense,   # (36,156,156)
            "input_low":    in_low,     # (12,156,156),
            "precipitation_targets": {
                "total_precipitation": high_precip_6h[-1]   # (128,128)
            },
            "surface_targets": {
                "temperature_2m": t2m_6h[-1],  # (32,32)
                "dewpoint_2m":    d2m_6h[-1],  # (32,32)
            },
            "hrrr_target": hrrr_36ch,   # (36,32,32)
        }
        return sample

# ============================================================
# (2) DataLoader
# ============================================================
train_root = r"/projects/aiid/KIPOT_SKT/Weather/trainset"
val_root   = r"/projects/aiid/KIPOT_SKT/Weather/validationset"
test_root  = r"/projects/aiid/KIPOT_SKT/Weather/testset"

train_dataset = WeatherBenchDataset(train_root)
val_dataset   = WeatherBenchDataset(val_root)
test_dataset  = WeatherBenchDataset(test_root)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,  num_workers=3)
val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False, num_workers=3)
test_loader  = DataLoader(test_dataset,  batch_size=2, shuffle=False, num_workers=3)

# ============================================================
# (3) MetNet3 모델 준비
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
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.device("cuda:1")

optimizer = torch.optim.Adam(metnet3.parameters(), lr=1e-4)

# ============================================================
# (5) Training Loop with Early Stopping (CRPS & CSI 기반)
# ============================================================
best_crps = float('inf')   # CRPS는 낮을수록 좋음
best_csi  = 0.0            # CSI는 높을수록 좋음
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
        # ----- 1. 데이터 준비 -----
        lead_times = batch['lead_time'].to(device)
        in_sparse  = batch['input_sparse'].to(device)
        in_stale   = batch['input_stale'].to(device)
        in_dense   = batch['input_dense'].to(device)
        in_low     = batch['input_low'].to(device)

        precip_targets = { k: v.to(device) for k,v in batch['precipitation_targets'].items() }
        surface_targets = { k: v.to(device) for k,v in batch['surface_targets'].items() }
        hrrr_target = batch['hrrr_target'].to(device)

        # ----- 2. Forward & Loss -----
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

    # ----- (A) Validation 스텝에서 CRPS와 CSI 계산 -----
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

            # ---------- (1) CRPS 계산 ----------
            #  - precipitation_preds: dict = { 'total_precipitation': (B, C, H, W) logits }
            precipitation_preds = pred.precipitation  # dict
            logits = precipitation_preds['total_precipitation']  # (B, C, H, W)
            probs = F.softmax(logits, dim=1)  # (B, C, H, W)

            # 타겟
            tgt = precip_targets['total_precipitation']  # (B, H, W)
            # CRPS
            crps_batch = compute_crps(probs, tgt)
            # ---------- (2) CSI 계산 ----------
            #  - 임계값 bin = 8, 확률 threshold=0.5 예시
            csi_batch = compute_csi(probs, tgt, threshold_bin=8, prob_threshold=0.5)

            total_crps += crps_batch
            total_csi += csi_batch
            num_batches += 1

    mean_crps = total_crps / num_batches
    mean_csi = total_csi / num_batches

    print(f"[Epoch {epoch}] Validation CRPS: {mean_crps:.4f}, CSI: {mean_csi:.4f}")

    # ----- (B) 개선 여부 체크 -----
    #     CRPS는 낮아질수록 좋고, CSI는 높아질수록 좋음.
    #     여기서는 "CRPS 개선 OR CSI 개선" 만으로 no_improve_count를 리셋하는 예시
    improved = False

    # CRPS가 더 낮다면
    if mean_crps < best_crps:
        best_crps = mean_crps
        improved = True

    # CSI가 더 높다면
    if mean_csi > best_csi:
        best_csi = mean_csi
        improved = True

    if improved:
        best_model_state = copy.deepcopy(metnet3.state_dict())
        no_improve_count = 0
        print(f"Improved CRPS or CSI.  => CRPS={best_crps:.4f}, CSI={best_csi:.4f}")
    else:
        no_improve_count += 1
        print(f"No improvement in CRPS or CSI for {no_improve_count} consecutive epoch(s).")

    # ----- (C) Early Stopping 로직 -----
    # min_epochs 이후, max_no_improve 연속 개선 없으면 정지
    if epoch >= min_epochs and no_improve_count >= max_no_improve:
        print(f"Early stopping after {epoch} epochs (no improvement for {no_improve_count} eps).")
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