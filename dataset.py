import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# Dataset 정의 (원본과 동일)
# ============================================================
class WeatherBenchDataset(Dataset):
    """
    개별 파일에 저장된 샘플들을 읽어와 MetNet3 입력 형태로 제공하는 Dataset.
    각 파일은 전처리된 샘플 하나를 저장하고 있으며, max_samples 파라미터로 사용할 샘플 개수를 제한.
    """
    def __init__(self, root_dir, max_samples=None):
        """
        root_dir 예:
        E:\metnet3\weather_bench\trainset
        max_samples: 사용할 샘플 개수를 지정, None인 경우 전체 파일을 사용.
        """
        self.root_dir = root_dir

        """
        Sorting 하는 이유는 같은 샘플을 보장하기 위함.
        """
        
        # =====================================
        # (1) 입력 (Input)경로 불러오기
        # =====================================
        self.input_sparse_paths = sorted(glob.glob(os.path.join(root_dir, 'input_sparse', 'input_sparse_sample_*.npy')))
        self.input_stale_paths  = sorted(glob.glob(os.path.join(root_dir, 'input_stale',  'input_stale_sample_*.npy')))
        self.input_dense_paths  = sorted(glob.glob(os.path.join(root_dir, 'input_dense',  'input_dense_sample_*.npy')))
        self.input_low_paths    = sorted(glob.glob(os.path.join(root_dir, 'input_low',    'input_low_sample_*.npy')))

        # =====================================
        # (2) 타겟 (Target)경로 불러오기
        # =====================================
        # (a) sparse_target (각 변수별 (N,6,32,32))
        self.t2m_paths      = sorted(glob.glob(os.path.join(root_dir, 'sparse_target', '2m_temperature_sample_*.npy')))
        self.d2m_paths      = sorted(glob.glob(os.path.join(root_dir, 'sparse_target', '2m_dewpoint_temperature_sample_*.npy')))
        self.precip_paths   = sorted(glob.glob(os.path.join(root_dir, 'sparse_target', 'total_precipitation_sample_*.npy')))

        # (b) dense_target = (N,36,32,32)
        self.dense_target_paths = sorted(glob.glob(os.path.join(root_dir, 'dense_target', 'dense_target_sample_*.npy')))

        # (c) high_target 예: total_precipitation (N,6,128,128)
        self.high_precip_paths  = sorted(glob.glob(os.path.join(root_dir, 'high_target', 'total_precipitation_sample_*.npy')))


        # =====================================
        # (3)  max_samples 갯수만큼 샘플 슬라이스
        # =====================================
        
        # (a) 총 샘플 수는 input_sparse 파일 수 기준
        self.num_samples = len(self.input_sparse_paths)

        # (b) max_samples가 지정된 경우 각 파일 리스트를 해당 개수만큼 슬라이스.
        if max_samples is not None:
            self.input_sparse_paths = self.input_sparse_paths[:max_samples]
            self.input_stale_paths  = self.input_stale_paths[:max_samples]
            self.input_dense_paths  = self.input_dense_paths[:max_samples]
            self.input_low_paths    = self.input_low_paths[:max_samples]
            self.t2m_paths          = self.t2m_paths[:max_samples]
            self.d2m_paths          = self.d2m_paths[:max_samples]
            self.precip_paths       = self.precip_paths[:max_samples]
            self.dense_target_paths = self.dense_target_paths[:max_samples]
            self.high_precip_paths  = self.high_precip_paths[:max_samples]
            
            

        # ==========================
        # (3) Bin Names 정의
        # ==========================
        self.surface_bin_names = ('temperature_2m', 'dewpoint_2m')
        self.precipitation_bin_names = ('total_precipitation',)



    def __len__(self):
        return self.num_samples

    # np.load 후 torch.from_numpy로 변환
    def load_tensor(self, path, dtype=torch.float): 
        return torch.from_numpy(np.load(path)).to(dtype)

    def __getitem__(self, idx):
        # 입력 데이터 개별 파일 로드
        
        in_sparse = self.load_tensor(self.input_sparse_paths[idx], dtype=torch.float)
        in_stale  = self.load_tensor(self.input_stale_paths[idx],  dtype=torch.float)
        in_dense  = self.load_tensor(self.input_dense_paths[idx],  dtype=torch.float)
        in_low    = self.load_tensor(self.input_low_paths[idx],    dtype=torch.float)

        t2m       = self.load_tensor(self.t2m_paths[idx],      dtype=torch.long)
        d2m       = self.load_tensor(self.d2m_paths[idx],      dtype=torch.long)
        precip    = self.load_tensor(self.precip_paths[idx],   dtype=torch.long)
        dense_target = self.load_tensor(self.dense_target_paths[idx], dtype=torch.long)
        high_precip  = self.load_tensor(self.high_precip_paths[idx],  dtype=torch.long)


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
                "total_precipitation": high_precip[-1]  # high_target의 마지막 프레임 사용
            },
            "surface_targets": {
                "temperature_2m": t2m[-1],
                "dewpoint_2m":    d2m[-1],
            },
            # HRRR-like 타겟
            "hrrr_target": dense_target  # HRRR-like 타겟
        }
        return sample
