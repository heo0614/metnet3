# %% 
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
# ============================================================
# (1) Dataset 정의 (사용자 코드 예시 그대로)
# ============================================================
def compute_crps(pred_probs: torch.Tensor, target_labels: torch.Tensor) -> float:
    """
    Discrete CRPS 계산을 위한 함수
    - pred_probs: (B, C, H, W) 예측 확률분포 (소프트맥스 결과)
    - target_labels: (B, H, W) 실제 정답 (클래스 인덱스)
    반환: CRPS (float)
    """
    B, C, H, W = pred_probs.shape
    assert target_labels.shape == (B, H, W)

    device = pred_probs.device

    # (1) 예측 CDF
    cdf_pred = torch.cumsum(pred_probs, dim=1)  # (B, C, H, W)
    # (2) 실제 CDF
    #   - target_labels == g 일 때, cdf_true[k] = 1(g <= k)
    cdf_true = (torch.arange(C, device=device).view(1, C, 1, 1) >= target_labels.unsqueeze(1)).float()

    # (3) 차이 제곱
    diff_sq = (cdf_pred - cdf_true) ** 2  # (B, C, H, W)
    # (4) 평균
    crps_val = diff_sq.mean().item()
    return crps_val

def compute_csi(pred_probs: torch.Tensor, target_labels: torch.Tensor,
                threshold_bin: int = 8,
                prob_threshold: float = 0.5) -> float:
    """
    CSI(Critical Success Index) 계산 함수
    - pred_probs: (B, C, H, W) 예측 확률분포
    - target_labels: (B, H, W) 실제 정답 (클래스 인덱스)
    - threshold_bin: 임의 예시 (e.g. 8) -> 8번 bin 이상을 '비가 온다'고 판단
    - prob_threshold: 예측 확률이 이 이상이면 양성(positive)으로 간주
    반환: CSI 스코어 (float)
    """
    B, C, H, W = pred_probs.shape
    device = pred_probs.device

    # (1) threshold_bin 이상이 될 확률 = sum_{k=threshold_bin..C-1} pred_probs
    p_rain = pred_probs[:, threshold_bin:, :, :].sum(dim=1)  # (B,H,W)

    # (2) 예측 이진화
    pred_positive = (p_rain >= prob_threshold)

    # (3) 실제 이진화
    real_positive = (target_labels >= threshold_bin)

    # (4) TP / (TP + FP + FN)
    tp = (pred_positive & real_positive).sum().item()
    fp = (pred_positive & (~real_positive)).sum().item()
    fn = ((~pred_positive) & real_positive).sum().item()

    denom = tp + fp + fn
    csi = tp / denom if denom > 0 else 0.0
    return csi

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


def visualize_inference(model, data_loader, device, num_samples=1):
    """
    저장된 모델로 추론을 수행하고, 예측 vs. 실제 타겟을 시각화 + CRPS/CSI 계산
    """
    model.eval()

    # 1) 테스트 로더에서 첫 번째 배치를 가져온다
    batch = next(iter(data_loader))

    # 2) 입력 데이터를 device로 이동
    lead_times = batch['lead_time'].to(device)
    in_sparse  = batch['input_sparse'].to(device)
    in_stale   = batch['input_stale'].to(device)
    in_dense   = batch['input_dense'].to(device)
    in_low     = batch['input_low'].to(device)

    # 3) 타겟(ground truth)
    #   - 여기서는 precipitation_targets['total_precipitation']만 사용
    precip_targets = { k: v.to(device) for k,v in batch['precipitation_targets'].items() }

    with torch.no_grad():
        # 4) 모델 예측 (강수량, 표면변수 등)
        preds = model(
            lead_times       = lead_times,
            hrrr_input_2496  = in_dense,
            hrrr_stale_state = in_stale,
            input_2496       = in_sparse,
            input_4996       = in_low,
        )

    # 5) 예측 중 'total_precipitation' 로짓(logits) 꺼내기
    precipitation_logits = preds.precipitation['total_precipitation']  # (B, C, H, W)
    # softmax 확률분포
    precipitation_probs = F.softmax(precipitation_logits, dim=1)      # (B, C, H, W)
    # 가장 확률이 높은 bin 인덱스(0~511)
    precipitation_pred_labels = torch.argmax(precipitation_probs, dim=1)  # (B, H, W)

    # 6) 실제 타겟 (여기서는 high_target의 마지막 프레임) -> shape=(B, H, W)
    precipitation_true_labels = precip_targets['total_precipitation']  # (B, H, W)

    # 6-1) 배치 전체에 대한 CRPS/CSI 계산
    crps_val = compute_crps(precipitation_probs, precipitation_true_labels)
    csi_val  = compute_csi(precipitation_probs, precipitation_true_labels,
                           threshold_bin=8, prob_threshold=0.5)
    print(f"[Batch CRPS = {crps_val:.4f}, CSI = {csi_val:.4f}]")

    # 7) 시각화 (예: num_samples=1~B)
    n_samples = min(num_samples, precipitation_pred_labels.shape[0])
    for i in range(n_samples):
        pred_2d = precipitation_pred_labels[i].cpu().numpy()  # (H, W)
        true_2d = precipitation_true_labels[i].cpu().numpy()  # (H, W)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        # (A) 예측
        im0 = axes[0].imshow(pred_2d, cmap='jet')
        axes[0].set_title(f"Predicted Precip (bin index)")
        plt.colorbar(im0, ax=axes[0])

        # (B) 실제값
        im1 = axes[1].imshow(true_2d, cmap='jet')
        axes[1].set_title(f"True Precip (bin index)")
        plt.colorbar(im1, ax=axes[1])

        plt.suptitle(f"Sample #{i} - total_precipitation")
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    # ------------------------------------------------------------
    # 1) 저장된 모델 로드
    # ------------------------------------------------------------
    from metnet3_original import MetNet3

    # 모델 정의 (학습 때와 동일한 설정)
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

    # 저장 경로
    save_dir = "/projects/aiid/KIPOT_SKT/Weather/test_outputs"
    save_path = os.path.join(save_dir, "metnet3_final.pth")

    # 모델 가중치 로드
    state_dict = torch.load(save_path, map_location=device)
    metnet3.load_state_dict(state_dict)
    metnet3.eval()

    # ------------------------------------------------------------
    # 2) Test DataLoader 준비 (이미 앞선 코드 예시와 동일)
    # ------------------------------------------------------------
    test_root = r"/projects/aiid/KIPOT_SKT/Weather/testset"
    test_dataset = WeatherBenchDataset(test_root)
    test_loader  = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

    # ------------------------------------------------------------
    # 3) 시각화
    # ------------------------------------------------------------
    visualize_inference(metnet3, test_loader, device, num_samples=6)



# %%
