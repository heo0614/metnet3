import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel

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

# ------------------------------------------------------------
# (A) finetuning용 모델 설정
# ------------------------------------------------------------
# * 기존 (precip 모델)과 동일한 구조
metnet3_finetune = MetNet3(
    dim = 512,
    # lead_time을 논문에서는 722까지 쓰지만, 예시이므로 원본에 맞춤
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
metnet3_finetune.to(device)

# ------------------------------------------------------------
# (B) 기존 강우량 모델(precip)에서 학습된 가중치 로드
# ------------------------------------------------------------
save_dir = "/projects/aiid/KIPOT_SKT/Weather/test_outputs"
save_path_precip = os.path.join(save_dir, "metnet3_final.pth")  # 기존 학습결과
checkpoint = torch.load(save_path_precip, map_location=device)
metnet3_finetune.load_state_dict(checkpoint, strict=True)

print("Loaded precipitation-oriented model weights.")

# ------------------------------------------------------------
# (C) 논문처럼 "topographical embedding"을 무효화하고 싶다면 (선택 사항)
# ------------------------------------------------------------
# 예: embedding 파라미터를 0으로 만들고 grad를 막아버림
if hasattr(metnet3_finetune, "topo_embeddings"):
    with torch.no_grad():
        metnet3_finetune.topo_embeddings.weight.zero_()
    metnet3_finetune.topo_embeddings.weight.requires_grad = False
    print("Disabled topographical embeddings by zeroing out weights.")

# ------------------------------------------------------------
# (D) Optimizer 준비 (finetune 단계)
# ------------------------------------------------------------
optimizer = torch.optim.Adam(metnet3_finetune.parameters(), lr=1e-5)  # 학습률은 줄이는 경우가 많음

# ============================================================
# Training Loop with Early Stopping (5 Epochs No Improvement)
# ============================================================
SURFACE_LOSS_SCALE = 100.0
HRRR_LOSS_WEIGHT   = 1.0   # 모델 내부도 10으로 곱하지만, 다시 여기서 조정할 수도 있음

max_epochs = 30
for epoch in range(1, max_epochs+1):
    metnet3_finetune.train()
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

        # (A) 원본 모델 forward
        total_loss, loss_breakdown = metnet3_finetune(
            lead_times            = lead_times,
            hrrr_input_2496       = in_dense,
            hrrr_stale_state      = in_stale,
            input_2496            = in_sparse,
            input_4996            = in_low,
            precipitation_targets = precip_targets,
            surface_targets       = surface_targets,
            hrrr_target           = hrrr_target,
        )
        # -> 여기서의 total_loss 는 이미 surface + precip + (hrrr * weight) 더해진 값

        # (B) 하지만 우리가 따로 정의하는 "외부 가중치"로 다시 합산
        #  loss_breakdown.surface => dict of surface losses
        #  loss_breakdown.precipitation => dict of precipitation losses
        #  loss_breakdown.hrrr => single float (MSE)
        surface_loss = sum(loss_breakdown.surface.values())  # dict values 합
        precip_loss  = sum(loss_breakdown.precipitation.values())
        hrrr_loss    = loss_breakdown.hrrr

        # (C) 원하는 비중으로 다시 합산 (논문처럼 surface=100배, HRRR=10배 등)
        # 만약 hrrr_loss_weight도 이 부분에서 곱하고 싶다면 아래처럼
        new_loss = (surface_loss * SURFACE_LOSS_SCALE) \
                   + (precip_loss) \
                   + (hrrr_loss * HRRR_LOSS_WEIGHT)

        # (D) backward
        new_loss.backward()
        optimizer.step()

        epoch_loss += new_loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"[Finetuning E{epoch}] Loss={avg_loss:.4f} (surface x {SURFACE_LOSS_SCALE})")

print("Done Finetuning for OMO with external weighting")
save_path_omo = "metnet3_final_omo.pth"
torch.save(metnet3_finetune.state_dict(), save_path_omo)
