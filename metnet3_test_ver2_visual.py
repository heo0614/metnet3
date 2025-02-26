# %% 
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_preprocessing import test_data_root
from dataset import WeatherBenchDataset


def visualize_inference(model, data_loader, device, num_samples=1):
    """
    저장된 모델로 추론을 수행하고, 예측 vs. 실제 타겟을 시각화하는 예시 함수
    num_samples: 시각화할 샘플 개수 (batch size 이상으로 지정해도 batch 범위 내로 제한됨)
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

    # 3) 타겟(ground truth)은 CPU로 바로 불러도 되고, 비교 편의상 GPU로 올릴 수도 있음
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

    # 7) 시각화 (예: num_samples=1개만 시각화)
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
        num_lead_times = 180,
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
    test_dataset = WeatherBenchDataset(test_data_root)
    test_loader  = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

    # ------------------------------------------------------------
    # 3) 시각화
    # ------------------------------------------------------------
    visualize_inference(metnet3, test_loader, device, num_samples=6)



# %%
