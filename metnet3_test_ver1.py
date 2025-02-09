import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
        self.input_sparse = np.load(os.path.join(root_dir, 'input_sparse.npy'))  # (N,30,156,156)
        self.input_stale  = np.load(os.path.join(root_dir, 'input_stale.npy'))   # (N, 6,156,156)
        self.input_dense  = np.load(os.path.join(root_dir, 'input_dense.npy'))   # (N,36,156,156)
        self.input_low    = np.load(os.path.join(root_dir, 'input_low.npy'))     # (N,12,156,156)

        # 샘플 수
        self.num_samples = self.input_sparse.shape[0]

        # ==========================
        # (2) 타겟 (Target) 불러오기
        # ==========================
        # (a) sparse_target (각 변수별 (N,6,32,32))
        self.t2m = np.load(os.path.join(root_dir, 'sparse_target', '2m_temperature.npy'))
        self.d2m = np.load(os.path.join(root_dir, 'sparse_target', '2m_dewpoint_temperature.npy'))
        self.precip32 = np.load(os.path.join(root_dir, 'sparse_target', 'total_precipitation.npy'))

        # (b) dense_target = (N,36,32,32)
        #   => 이미 "6시간 × 6변수"가 36채널로 플래튼 된 상태
        self.dense_target_36ch = np.load(os.path.join(root_dir, 'dense_target.npy'))  # (N,36,32,32)

        # (c) high_target 예: total_precipitation (N,6,128,128)
        self.high_precip = np.load(os.path.join(root_dir, 'high_target', 'total_precipitation.npy'))

        # (d) HRRR-like 타겟 (N,36,32,32)로 쓰고 싶다면
        #     여기서는 dense_target을 그대로 활용한다고 가정
        self.hrrr_36ch = self.dense_target_36ch  # shape = (N,36,32,32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1) 입력을 Torch 텐서로 변환 (float)
        in_sparse = torch.from_numpy(self.input_sparse[idx]).float()  # (30,156,156)
        in_stale  = torch.from_numpy(self.input_stale[idx]).float()   # ( 6,156,156)
        in_dense  = torch.from_numpy(self.input_dense[idx]).float()   # (36,156,156)
        in_low    = torch.from_numpy(self.input_low[idx]).float()     # (12,156,156)

        # 2) 타겟 변환
        #   (a) sparse_target : 6시간 (N,6,32,32)
        #       여기서는 예시로 "마지막 타임스텝"만 쓸 수도 있고,
        #       혹은 6시간 전부를 유지할 수도 있습니다.
        #       여기서는 "6시간 전부"를 예시로 보여줌
        t2m_6h     = torch.from_numpy(self.t2m[idx]).long()        # (6,32,32)
        d2m_6h     = torch.from_numpy(self.d2m[idx]).long()        # (6,32,32)
        precip_6h  = torch.from_numpy(self.precip32[idx]).long()   # (6,32,32)

        #   (b) dense_target (N,36,32,32) -> (36,32,32) 단일 프레임
        dense_target_36ch = torch.from_numpy(self.dense_target_36ch[idx]).long()  # (36,32,32)

        #   (c) high_target (N,6,128,128)
        high_precip_6h = torch.from_numpy(self.high_precip[idx]).long()           # (6,128,128)

        #   (d) HRRR-like : (N,36,32,32)
        hrrr_36ch = torch.from_numpy(self.hrrr_36ch[idx]).float()                 # (36,32,32)

        # 3) lead_time 예시 (0~721 중 하나)
        lead_time = torch.tensor(np.random.randint(0, 722), dtype=torch.long)  # Convert to tensor

        # 4) 딕셔너리 구성
        sample = {
            "lead_time": lead_time,

            # ---- 입력 ----
            "input_sparse": in_sparse,  # (30,156,156)
            "input_stale":  in_stale,   # ( 6,156,156)
            "input_dense":  in_dense,   # (36,156,156)
            "input_low":    in_low,     # (12,156,156)

            # ---- 타겟(예시) ----
            # precipitation_targets, surface_targets는
            # MetNet3 forward에서 요구하는 dict 구조를 맞춰 주시면 됩니다.
            "precipitation_targets": {
                # 예: high_target를 씀 (마지막 타임스텝 등으로 가공 가능)
                # 여기서는 6시간 모두를 (B,6,128,128)로 넣어도 되지만
                # 원래 MetNet3는 (B,128,128) 단일 타임 예측 구조이므로
                # 필요 시 "마지막 step"만 취하는 식으로 수정 가능
                "total_precipitation": high_precip_6h[-1]  # (128,128) 예시
            },
            "surface_targets": {
                "temperature_2m": t2m_6h[-1],   # (32,32)
                "dewpoint_2m":    d2m_6h[-1],   # (32,32)
                # wind_u_10m, wind_v_10m, land_sea_mask 등도
                # sparse_target에 있으면 여기서 추가
            },
            # hrrr_target 예: (36,32,32)
            "hrrr_target": hrrr_36ch
        }

        # (참고) 만약 6시간 전체를 넣고 싶다면
        # sample["precipitation_targets"]["total_precipitation"] = high_precip_6h  # (6,128,128)
        # 와 같이 바꾸고, 모델 내부 로직(멀티 타임스텝)을 수정해야 합니다.

        return sample


# ==========================
# Data Loading
# ==========================

train_root = r"/projects/aiid/KIPOT_SKT/Weather/trainset"
val_root   = r"/projects/aiid/KIPOT_SKT/Weather/validationset"
test_root  = r"/projects/aiid/KIPOT_SKT/Weather/testset"

train_dataset = WeatherBenchDataset(train_root)
val_dataset   = WeatherBenchDataset(val_root)
test_dataset  = WeatherBenchDataset(test_root)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False, num_workers=4)

# ==========================
# Model Initialization
# ==========================

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

    # 채널
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
        # wind_u_10m     = 256,
        # wind_v_10m     = 256,
        # land_sea_mask  = 2,
    ),

    hrrr_loss_weight = 10,
    hrrr_norm_strategy = 'sync_batchnorm',
    hrrr_norm_statistics = None,

    crop_size_post_16km = 32,
    resnet_block_depth = 2,
)

# ==========================
# Device Configuration
# ==========================

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move the model to the device
metnet3.to(device)

# ==========================
# Optimizer
# ==========================

optimizer = torch.optim.Adam(metnet3.parameters(), lr=1e-4)

# ==========================
# Training Loop
# ==========================

num_epochs = 2
metnet3.train()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in train_loader:
        # (A) 입력
        lead_times = batch['lead_time'].to(device)   # (B,)
        in_sparse  = batch['input_sparse'].to(device) # (B,30,156,156)
        in_stale   = batch['input_stale'].to(device)  # (B, 6,156,156)
        in_dense   = batch['input_dense'].to(device)  # (B,36,156,156)
        in_low     = batch['input_low'].to(device)    # (B,12,156,156)

        # (B) 타겟
        # Move each target tensor to the device
        precip_targets = {
            key: value.to(device)
            for key, value in batch['precipitation_targets'].items()
        }
        surface_targets = {
            key: value.to(device)
            for key, value in batch['surface_targets'].items()
        }
        hrrr_target = batch['hrrr_target'].to(device)   # (B,36,32,32)

        # (C) 모델 forward
        total_loss, loss_breakdown = metnet3(
            lead_times            = lead_times,
            hrrr_input_2496      = in_dense,
            hrrr_stale_state     = in_stale,
            input_2496           = in_sparse,
            input_4996           = in_low,
            precipitation_targets= precip_targets,
            surface_targets      = surface_targets,
            hrrr_target          = hrrr_target,
        )

        # (D) backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{num_epochs}] Average Loss = {avg_epoch_loss:.4f}, Breakdown={loss_breakdown}")

# ==========================
# Evaluation Loop
# ==========================

metnet3.eval()
with torch.no_grad():
    for batch in test_loader:
        lead_times = batch['lead_time'].to(device)
        in_sparse  = batch['input_sparse'].to(device)
        in_stale   = batch['input_stale'].to(device)
        in_dense   = batch['input_dense'].to(device)
        in_low     = batch['input_low'].to(device)

        # 모델에 타겟 없이 호출 => 예측값 3가지 리턴
        surface_preds, hrrr_pred, precipitation_preds = metnet3(
            lead_times         = lead_times,
            hrrr_input_2496    = in_dense,
            hrrr_stale_state   = in_stale,
            input_2496         = in_sparse,
            input_4996         = in_low,
        )

        # surface_preds: Dict[str, Tensor], e.g. {'temperature_2m': (B,32,32), ...}
        # hrrr_pred    : Tensor, shape=(B, 36, 32, 32)
        # precipitation_preds: Dict[str, Tensor], e.g. {'total_precipitation': (B,128,128)}

        # 필요 시 후처리(softmax -> argmax 등) 후 저장/평가
        # 예시이므로 첫 배치만 확인 후 종료
        break
