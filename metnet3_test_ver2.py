import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Dataset 정의 (원본과 동일)
# ============================================================
class WeatherBenchDataset(Dataset):
    """
    전처리된 .npy (6시간 입력 + 6시간 타겟) 파일을 로드하여
    MetNet3에 들어갈 형태로 제공하는 예시 Dataset
    """
    def __init__(self, root_dir):
        """
        root_dir 예:
          E:\metnet3\weather_bench\trainset
        """
        self.root_dir = root_dir

        # ==========================
        # (1) 입력 (Input) 불러오기
        # ==========================
        self.input_sparse = np.load(os.path.join(root_dir, 'input_sparse_normalized.npy'))  # (N,30,156,156)
        self.input_stale  = np.load(os.path.join(root_dir, 'input_stale_normalized.npy'))   # (N, 6,156,156)
        self.input_dense  = np.load(os.path.join(root_dir, 'input_dense_normalized.npy'))   # (N,36,156,156)
        self.input_low    = np.load(os.path.join(root_dir, 'input_low_normalized.npy'))     # (N,12,156,156)

        self.num_samples = self.input_sparse.shape[0]

        # ==========================
        # (2) 타겟 (Target) 불러오기
        # ==========================
        # (a) sparse_target (각 변수별 (N,6,32,32))
        self.t2m = np.load(os.path.join(root_dir, 'sparse_target', '2m_temperature.npy'))
        self.d2m = np.load(os.path.join(root_dir, 'sparse_target', '2m_dewpoint_temperature.npy'))
        self.precip32 = np.load(os.path.join(root_dir, 'sparse_target', 'total_precipitation.npy'))

        # (b) dense_target = (N,36,32,32)
        self.dense_target_36ch = np.load(os.path.join(root_dir, 'dense_target.npy'))  # (N,36,32,32)

        # (c) high_target 예: total_precipitation (N,6,128,128)
        self.high_precip = np.load(os.path.join(root_dir, 'high_target', 'total_precipitation.npy'))

        # (d) HRRR-like 타겟 (N,36,32,32)
        self.hrrr_36ch = self.dense_target_36ch

        # ==========================
        # (3) Bin Names 정의
        # ==========================
        self.surface_bin_names = ('temperature_2m', 'dewpoint_2m')
        self.precipitation_bin_names = ('total_precipitation',)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 입력 텐서 변환 (float)
        in_sparse = torch.from_numpy(self.input_sparse[idx]).float()  # (30,156,156)
        in_stale  = torch.from_numpy(self.input_stale[idx]).float()   # ( 6,156,156)
        in_dense  = torch.from_numpy(self.input_dense[idx]).float()   # (36,156,156)
        in_low    = torch.from_numpy(self.input_low[idx]).float()     # (12,156,156)

        # 타겟 변환
        t2m_6h     = torch.from_numpy(self.t2m[idx]).long()        # (6,32,32)
        d2m_6h     = torch.from_numpy(self.d2m[idx]).long()        # (6,32,32)
        precip_6h  = torch.from_numpy(self.precip32[idx]).long()     # (6,32,32)
        dense_target_36ch = torch.from_numpy(self.dense_target_36ch[idx]).long()  # (36,32,32)
        high_precip_6h = torch.from_numpy(self.high_precip[idx]).long()           # (6,128,128)
        hrrr_36ch = torch.from_numpy(self.hrrr_36ch[idx]).float()                 # (36,32,32)

        # lead_time 예시 (0~721 중 하나)
        lead_time = torch.tensor(np.random.randint(0, 722), dtype=torch.long)

        sample = {
            "lead_time": lead_time,
            # ---- 입력 ----
            "input_sparse": in_sparse,
            "input_stale":  in_stale,
            "input_dense":  in_dense,
            "input_low":    in_low,
            # ---- 타겟 (예시) ----
            "precipitation_targets": {
                # 여기서는 high_target의 마지막 프레임만 사용 (예: (128,128))
                "total_precipitation": high_precip_6h[-1]
            },
            "surface_targets": {
                "temperature_2m": t2m_6h[-1],
                "dewpoint_2m":    d2m_6h[-1],
            },
            # HRRR-like 타겟
            "hrrr_target": hrrr_36ch
        }
        return sample

# ============================================================
# Data Loading
# ============================================================
train_root = r"/projects/aiid/KIPOT_SKT/Weather/trainset"
val_root   = r"/projects/aiid/KIPOT_SKT/Weather/validationset"
test_root  = r"/projects/aiid/KIPOT_SKT/Weather/testset"

train_dataset = WeatherBenchDataset(train_root)
val_dataset   = WeatherBenchDataset(val_root)
test_dataset  = WeatherBenchDataset(test_root)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False, num_workers=4)

# ============================================================
# Model Initialization
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
    hrrr_channels = 36,      # (dense=36채널)
    input_2496_channels = 30,# (sparse=30채널)
    input_4996_channels = 12,# (low=12채널)
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
print(f"Using device: {device}")
metnet3.to(device)

optimizer = torch.optim.Adam(metnet3.parameters(), lr=1e-4)

# ============================================================
# Training Loop with Early Stopping (최소 30 에폭 진행 후 5 에폭 연속 개선 없으면 종료)
# ============================================================
best_precip_acc = 0.0
best_model_state = None
min_epochs = 30       # 최소 에폭 수
max_no_improve = 5    # 개선 없는 에폭이 5회 연속이면 종료
epoch = 0
no_improve_count = 0  # 연속 개선이 없는 에폭 수

while True:
    epoch += 1
    metnet3.train()
    epoch_loss = 0.0
    for batch in train_loader:
        # (A) 입력 데이터 준비
        lead_times = batch['lead_time'].to(device)   # (B,)
        in_sparse  = batch['input_sparse'].to(device) # (B,30,156,156)
        in_stale   = batch['input_stale'].to(device)  # (B,6,156,156)
        in_dense   = batch['input_dense'].to(device)  # (B,36,156,156)
        in_low     = batch['input_low'].to(device)    # (B,12,156,156)

        # (B) 타겟 (precipitation, surface, HRRR)
        precip_targets = { key: value.to(device) for key, value in batch['precipitation_targets'].items() }
        surface_targets = { key: value.to(device) for key, value in batch['surface_targets'].items() }
        hrrr_target = batch['hrrr_target'].to(device)   # (B,36,32,32)

        # (C) 모델 forward 및 loss 계산
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
    print(f"[Epoch {epoch}] Average Training Loss = {avg_epoch_loss:.4f}, Loss Breakdown = {loss_breakdown}")

    # ============================================================
    # Validation: Precipitation Accuracy만 계산
    # ============================================================
    metnet3.eval()
    total_correct = 0
    total_pixels = 0
    with torch.no_grad():
        for batch in val_loader:
            lead_times = batch['lead_time'].to(device)
            in_sparse  = batch['input_sparse'].to(device)
            in_stale   = batch['input_stale'].to(device)
            in_dense   = batch['input_dense'].to(device)
            in_low     = batch['input_low'].to(device)
            precip_targets = { key: value.to(device) for key, value in batch['precipitation_targets'].items() }

            # 모델 forward (타겟 없이 예측)
            pred = metnet3(
                lead_times       = lead_times,
                hrrr_input_2496  = in_dense,
                hrrr_stale_state = in_stale,
                input_2496       = in_sparse,
                input_4996       = in_low,
            )
            # precipitation 예측만 사용 (dict: key는 'total_precipitation')
            precipitation_preds = pred.precipitation
            for key, logits in precipitation_preds.items():
                probs = F.softmax(logits, dim=1)       # (B, C, H, W)
                preds = torch.argmax(probs, dim=1)       # (B, H, W)
                pred_labels = preds.reshape(-1)
                target_labels = precip_targets[key].cpu().numpy().reshape(-1)
                correct = (pred_labels.cpu().numpy() == target_labels).sum()
                total_correct += correct
                total_pixels += target_labels.size

    precip_acc = total_correct / total_pixels if total_pixels > 0 else 0
    print(f"[Epoch {epoch}] Validation Precipitation Accuracy: {precip_acc:.4f}")

    # 개선이 있을 경우 best model 저장, 없으면 no_improve_count 증가
    if precip_acc > best_precip_acc:
        best_precip_acc = precip_acc
        best_model_state = copy.deepcopy(metnet3.state_dict())
        no_improve_count = 0  # 개선 시 카운터 리셋
        print(f"Improved precipitation accuracy to {best_precip_acc:.4f}. Continuing training.")
    else:
        no_improve_count += 1
        print(f"No improvement in precipitation accuracy for {no_improve_count} consecutive epoch(s).")

    # 최소 min_epochs 이후 5 에폭 연속 개선 없으면 조기 종료
    if epoch >= min_epochs and no_improve_count >= max_no_improve:
        print(f"Early stopping triggered after {epoch} epochs with {no_improve_count} consecutive epochs without improvement.")
        break

# ============================================================
# 최종 모델 저장 (향상된 best model을 저장)
# ============================================================
if best_model_state is not None:
    metnet3.load_state_dict(best_model_state)

save_dir = "/projects/aiid/KIPOT_SKT/Weather/test_outputs"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "metnet3_final.pth")
torch.save(metnet3.state_dict(), save_path)
print(f"Final model saved to {save_path}")
