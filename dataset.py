import os

import numpy as np
import torch
from torch.utils.data import Dataset


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
