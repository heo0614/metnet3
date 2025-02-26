import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_preprocessing import train_data_root, valid_data_root, test_data_root
from dataset import WeatherBenchDataset

if __name__ == '__main__':
    # ============================================================
    # Data Loading
    # ============================================================
    train_dataset = WeatherBenchDataset(train_data_root)
    val_dataset   = WeatherBenchDataset(valid_data_root)
    test_dataset  = WeatherBenchDataset(test_data_root)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False, num_workers=4)

    # ============================================================
    # Model Initialization
    # ============================================================
    from metnet3_original import MetNet3

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
    torch.device("cuda:1")

    optimizer = torch.optim.Adam(metnet3.parameters(), lr=1e-4)

    # ============================================================
    # Training Loop with Early Stopping (최소 30 에폭 진행 후 5 에폭 연속 개선 없으면 종료)
    # ============================================================
    # ============================================================
    # Training Loop with Early Stopping (정확도가 향상되면 5번 연속 개선 없을 때까지 학습)
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
            pred, total_loss, loss_breakdown = metnet3(
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

            # # visualize
            # # 5) 예측 중 'total_precipitation' 로짓(logits) 꺼내기
            # precipitation_logits = preds.precipitation['total_precipitation']  # (B, C, H, W)
            # # softmax 확률분포
            # precipitation_probs = F.softmax(precipitation_logits, dim=1)  # (B, C, H, W)
            # # 가장 확률이 높은 bin 인덱스(0~511)
            # precipitation_pred_labels = torch.argmax(precipitation_probs, dim=1)  # (B, H, W)
            #
            # # 6) 실제 타겟 (여기서는 high_target의 마지막 프레임) -> shape=(B, H, W)
            # precipitation_true_labels = precip_targets['total_precipitation']  # (B, H, W)
            # pred_2d = precipitation_pred_labels[0].cpu().numpy()  # (H, W)
            # true_2d = precipitation_true_labels[0].cpu().numpy()  # (H, W)
            #
            # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            # # (A) 예측
            # im0 = axes[0].imshow(pred_2d, cmap='jet')
            # axes[0].set_title(f"Predicted Precip (bin index)")
            # plt.colorbar(im0, ax=axes[0])
            #
            # # (B) 실제값
            # im1 = axes[1].imshow(true_2d, cmap='jet')
            # axes[1].set_title(f"True Precip (bin index)")
            # plt.colorbar(im1, ax=axes[1])
            #
            # plt.suptitle(f"Sample - total_precipitation")
            # plt.tight_layout()
            # plt.savefig('temp_plot.png', bbox_inches='tight')

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

        # 개선이 없으면 no_improve_count 리셋
        if no_improve_count == max_no_improve:
            print(f"No improvement for {max_no_improve} consecutive epochs. Restarting the training.")
            no_improve_count = 0  # 정확도가 향상되면 카운트를 리셋하고 다시 5번 동안 정확도 향상이 없을 때까지 학습을 진행합니다.

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

