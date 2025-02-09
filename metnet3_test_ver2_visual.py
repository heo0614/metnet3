# %% 
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
from metnet3_original import MetNet3  # 모델 클래스 임포트

class WeatherBenchDataset(Dataset):
    """
    예: 전처리된 .npy에서 6시간(시간축=6) 입력 + 6시간 타겟 불러오기
    여기서는 sparse_target만 사용 (dense_target도 로드하지만 시각화 X).
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # ---------- (A) 입력 ----------
        self.input_sparse = np.load(os.path.join(root_dir, 'input_sparse_normalized.npy'))  # (N,30,156,156)
        self.input_stale  = np.load(os.path.join(root_dir, 'input_stale_normalized.npy'))   # (N,6,156,156)
        self.input_dense  = np.load(os.path.join(root_dir, 'input_dense_normalized.npy'))   # (N,36,156,156)
        self.input_low    = np.load(os.path.join(root_dir, 'input_low_normalized.npy'))     # (N,12,156,156)
        self.num_samples  = self.input_sparse.shape[0]

        # ---------- (B) Sparse 타겟 (N,6,32,32) ----------
        self.sparse_precip = np.load(
            os.path.join(root_dir, 'sparse_target', 'total_precipitation.npy')
        )
        # (C) Dense나 high target도 있겠지만, 여기서는 사용 X.

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        in_sparse = torch.from_numpy(self.input_sparse[idx]).float()
        in_stale  = torch.from_numpy(self.input_stale[idx]).float()
        in_dense  = torch.from_numpy(self.input_dense[idx]).float()
        in_low    = torch.from_numpy(self.input_low[idx]).float()

        # sparse_target
        sparse_precip_6h = torch.from_numpy(self.sparse_precip[idx]).long()  # (6,32,32)

        sample = {
            "input_sparse": in_sparse,
            "input_stale":  in_stale,
            "input_dense":  in_dense,
            "input_low":    in_low,

            "sparse_precip_6h": sparse_precip_6h,
        }
        return sample

# -----------------------------------------------------------
# 모델 로드 (예: 이미 학습된 checkpoint)
# -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MetNet3(
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
    hrrr_channels = 36,
    input_2496_channels = 30,
    input_4996_channels = 12,
    surface_and_hrrr_target_spatial_size = 32,
    precipitation_target_bins = {
        # sparse_precipitation => 256 bin (가정)
        "sparse_precipitation": 256
    },
    surface_target_bins = {
        "temperature_2m": 256,
        "dewpoint_2m":    256,
    },
    hrrr_loss_weight = 10,
    hrrr_norm_strategy = 'sync_batchnorm',
    hrrr_norm_statistics = None,
    crop_size_post_16km = 32,
    resnet_block_depth = 2,
)

checkpoint_path = "/projects/aiid/KIPOT_SKT/Weather/test_outputs/metnet3_final.pth"
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# -----------------------------------------------------------
# 시각화 함수: sparse target만 (GT vs Pred)
# -----------------------------------------------------------
def visualize_sparse_target():
    test_root = "/projects/aiid/KIPOT_SKT/Weather/testset"
    ds = WeatherBenchDataset(test_root)
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    batch = next(iter(loader))

    # 입력
    in_sparse = batch["input_sparse"].to(device)
    in_stale  = batch["input_stale"].to(device)
    in_dense  = batch["input_dense"].to(device)
    in_low    = batch["input_low"].to(device)

    # sparse 타겟(6,32,32)
    sparse_precip_6h = batch["sparse_precip_6h"]  # shape=(1,6,32,32)

    plt.figure(figsize=(12, 12))

    # 6시간 loop
    for i in range(6):
        # lead_time: i=0=>1h(30), i=1=>2h(60) ...
        lead_time_val = (i+1)*30
        lt_tensor = torch.tensor([lead_time_val], dtype=torch.long, device=device)

        with torch.no_grad():
            output = model(
                lead_times       = lt_tensor,
                hrrr_input_2496  = in_dense,
                hrrr_stale_state = in_stale,
                input_2496       = in_sparse,
                input_4996       = in_low,
            )

        # sparse precip 예측 로짓 => softmax => argmax
        logits_sparse = output.precipitation["sparse_precipitation"]  # shape=(1,256,32,32)
        probs_sparse  = F.softmax(logits_sparse, dim=1)
        pred_bin_sparse = torch.argmax(probs_sparse, dim=1)  # (1,32,32)

        # (a) GT (i번째 시간)
        gt_i = sparse_precip_6h[:, i]  # shape=(1,32,32)
        arr_gt = gt_i[0].cpu().numpy() # (32,32)

        # (b) Pred
        pred_i = pred_bin_sparse[0].cpu().numpy() # (32,32)

        # subplot(6행,2열)
        row_gt  = 2*i + 1
        row_pred= 2*i + 2

        # GT
        plt.subplot(6,2, row_gt)
        plt.imshow(arr_gt, cmap='viridis')
        plt.title(f"Sparse GT hour={i+1} (lt={lead_time_val})")
        plt.colorbar()

        # Pred
        plt.subplot(6,2, row_pred)
        plt.imshow(pred_i, cmap='viridis')
        plt.title(f"Sparse Pred hour={i+1}")
        plt.colorbar()

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    visualize_sparse_target()


# %%
