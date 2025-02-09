import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class WeatherBenchDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # ============ 입력 ============
        self.input_sparse = np.load(os.path.join(root_dir, 'input_sparse.npy'))
        self.input_stale  = np.load(os.path.join(root_dir, 'input_stale.npy'))
        self.input_dense  = np.load(os.path.join(root_dir, 'input_dense.npy'))
        self.input_low    = np.load(os.path.join(root_dir, 'input_low.npy'))

        self.num_samples = self.input_sparse.shape[0]

        # ============ 타겟 ============
        self.t2m      = np.load(os.path.join(root_dir, 'sparse_target', '2m_temperature.npy'))
        self.d2m      = np.load(os.path.join(root_dir, 'sparse_target', '2m_dewpoint_temperature.npy'))
        self.precip32 = np.load(os.path.join(root_dir, 'sparse_target', 'total_precipitation.npy'))

        self.dense_target_36ch = np.load(os.path.join(root_dir, 'dense_target.npy'))   # (N,36,32,32)
        self.high_precip       = np.load(os.path.join(root_dir, 'high_target', 'total_precipitation.npy'))
        self.hrrr_36ch         = self.dense_target_36ch

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        in_sparse = torch.from_numpy(self.input_sparse[idx]).float()
        in_stale  = torch.from_numpy(self.input_stale[idx]).float()
        in_dense  = torch.from_numpy(self.input_dense[idx]).float()
        in_low    = torch.from_numpy(self.input_low[idx]).float()

        t2m_6h      = torch.from_numpy(self.t2m[idx]).long()        # (6,32,32)
        d2m_6h      = torch.from_numpy(self.d2m[idx]).long()        # (6,32,32)
        precip_6h   = torch.from_numpy(self.precip32[idx]).long()   # (6,32,32)
        dense_36ch  = torch.from_numpy(self.dense_target_36ch[idx]).long()  # (36,32,32)
        high_precip = torch.from_numpy(self.high_precip[idx]).long()        # (6,128,128)
        hrrr_36ch   = torch.from_numpy(self.hrrr_36ch[idx]).float()         # (36,32,32)

        lead_time = np.random.randint(0, 722)

        sample = {
            "lead_time": lead_time,
            # 입력
            "input_sparse": in_sparse,
            "input_stale":  in_stale,
            "input_dense":  in_dense,
            "input_low":    in_low,
            # 타겟 예시
            "precipitation_targets": {"total_precipitation": high_precip[-1]},
            "surface_targets": {
                "temperature_2m": t2m_6h[-1],
                "dewpoint_2m":    d2m_6h[-1],
            },
            "hrrr_target": hrrr_36ch  # (36,32,32)
        }
        return sample


# =================== DataLoader ===================
test_root = r"/projects/aiid/KIPOT_SKT/Weather/testset"
test_dataset = WeatherBenchDataset(test_root)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# =================== MetNet3 불러오기 ===================
from metnet3_original import MetNet3

metnet3 = MetNet3(
    dim=512,
    num_lead_times=722,
    lead_time_embed_dim=32,
    input_spatial_size=156,
    attn_depth=12,
    attn_dim_head=8,
    attn_heads=32,
    attn_dropout=0.1,
    vit_window_size=8,
    vit_mbconv_expansion_rate=4,
    vit_mbconv_shrinkage_rate=0.25,

    hrrr_channels=36,
    input_2496_channels=30,
    input_4996_channels=12,
    surface_and_hrrr_target_spatial_size=32,

    precipitation_target_bins=dict(total_precipitation=512),
    surface_target_bins=dict(
        temperature_2m=256,
        dewpoint_2m=256
    ),

    hrrr_loss_weight=10,
    hrrr_norm_strategy='sync_batchnorm',
    hrrr_norm_statistics=None,
    crop_size_post_16km=32,
    resnet_block_depth=2,
)

model_path = r"/projects/aiid/KIPOT_SKT/Weather/metnet3_final.pt"
state_dict = torch.load(model_path, map_location="cpu")
metnet3.load_state_dict(state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metnet3.to(device)
metnet3.eval()

output_dir = r"/projects/aiid/KIPOT_SKT/Weather/test_outputs"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        lead_times = batch['lead_time'].to(device)
        in_sparse  = batch['input_sparse'].to(device)
        in_stale   = batch['input_stale'].to(device)
        in_dense   = batch['input_dense'].to(device)
        in_low     = batch['input_low'].to(device)

        # 모델 추론
        surface_preds, hrrr_pred, precipitation_preds = metnet3(
            lead_times=lead_times,
            hrrr_input_2496=in_dense,
            hrrr_stale_state=in_stale,
            input_2496=in_sparse,
            input_4996=in_low,
        )

        # 리드타임 저장
        lead_times_np = lead_times.cpu().numpy()
        np.save(os.path.join(output_dir, f"batch_{batch_idx}_lead_times.npy"), lead_times_np)

        # ----- (1) surface_preds -----
        for var_name, pred_tensor in surface_preds.items():
            # 보통 (B, H, W) or (B,1,H,W) 형태
            pred_np = pred_tensor.cpu().numpy()
            np.save(os.path.join(output_dir, f"batch_{batch_idx}_surface_{var_name}.npy"), pred_np)

        # ----- (2) HRRR 예측 (B,36,32,32) -> (B,6,6,32,32) -----
        hrrr_np = hrrr_pred.cpu().numpy()  # (B,36,32,32)
        # reshape => (B,6,6,32,32) => 시간(6) × 채널(6) × 32×32
        B, CH, H, W = hrrr_np.shape
        if CH == 36:
            hrrr_np = hrrr_np.reshape(B, 6, 6, H, W)  # (B,6,6,32,32)

        np.save(os.path.join(output_dir, f"batch_{batch_idx}_hrrr_pred.npy"), hrrr_np)

        # ----- (3) 강수 예측(precipitation_preds) -----
        # 예) {"total_precipitation": (B, H, W)} 등
        for var_name, pred_tensor in precipitation_preds.items():
            pred_np = pred_tensor.cpu().numpy()
            np.save(os.path.join(output_dir, f"batch_{batch_idx}_precip_{var_name}.npy"), pred_np)

        print(f"[Batch {batch_idx}] 저장 완료.")

print("테스트셋 추론 완료 및 .npy 파일 저장이 모두 끝났습니다.")