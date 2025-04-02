import torch
import numpy as np
import random
from typing import Tuple, Optional

class TorchKriging:
    def __init__(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        values: torch.Tensor,
        variogram_model: str = 'exponential',
        sill: Optional[float] = None,
        range_param: Optional[float] = None,
        nugget: float = 0.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.x = x.to(device)
        self.y = y.to(device)
        self.values = values.to(device)
        self.variogram_model = variogram_model
        self.nugget = nugget
        self.dist_matrix = self._compute_distance_matrix(x, y)

        if sill is None or range_param is None:
            self.sill, self.range_param = self._estimate_variogram_params()
        else:
            self.sill = sill
            self.range_param = range_param

        self.kriging_matrix = self._build_kriging_matrix()
        extended_values = torch.cat([self.values.to(torch.float32), 
                             torch.tensor([0.0], device=self.device, dtype=torch.float32)])
        # print("Kriging matrix dtype:", self.kriging_matrix.dtype)
        # print("Extended values dtype:", extended_values.dtype)

        self.weights = torch.linalg.solve(self.kriging_matrix, extended_values) 

    def _compute_distance_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dx = x.unsqueeze(0) - x.unsqueeze(1)
        dy = y.unsqueeze(0) - y.unsqueeze(1)
        return torch.sqrt(dx**2 + dy**2)

    def _variogram_model_fn(self, h: torch.Tensor) -> torch.Tensor:
        if self.variogram_model == 'exponential':
            return self.sill * (1 - torch.exp(-h / self.range_param))
        elif self.variogram_model == 'gaussian':
            return self.sill * (1 - torch.exp(-(h**2) / (self.range_param**2)))
        elif self.variogram_model == 'spherical':
            h_scaled = h / self.range_param
            mask = h_scaled <= 1
            result = torch.zeros_like(h)
            result[mask] = self.sill * (1.5 * h_scaled[mask] - 0.5 * h_scaled[mask]**3)
            result[~mask] = self.sill
            return result
        else:
            raise ValueError(f"Unknown variogram model: {self.variogram_model}")

    def _estimate_variogram_params(self) -> Tuple[float, float]:
        sill = torch.var(self.values).item()
        range_param = torch.median(self.dist_matrix[self.dist_matrix > 0]).item()
        return sill, range_param

    def _build_kriging_matrix(self) -> torch.Tensor:
        n = len(self.x)
        variogram_vals = self._variogram_model_fn(self.dist_matrix)
        variogram_vals.diagonal().add_(self.nugget)
        extended_matrix = torch.zeros((n + 1, n + 1), device=self.device)
        extended_matrix[:n, :n] = variogram_vals
        extended_matrix[:n, n] = 1.0
        extended_matrix[n, :n] = 1.0
        return extended_matrix

    def predict(self, x_new: torch.Tensor, y_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_new = x_new.to(self.device)
        y_new = y_new.to(self.device)
        dx = x_new.unsqueeze(1) - self.x
        dy = y_new.unsqueeze(1) - self.y
        distances = torch.sqrt(dx**2 + dy**2)
        variogram_vals = self._variogram_model_fn(distances)
        ones = torch.ones((len(x_new), 1), device=self.device)
        variogram_extended = torch.cat([variogram_vals, ones], dim=1)
        predictions = torch.matmul(variogram_extended, self.weights)
        variances = self._compute_variances(variogram_extended)
        return predictions, variances

    def _compute_variances(self, variogram_extended: torch.Tensor) -> torch.Tensor:
        return torch.sum(variogram_extended * torch.matmul(variogram_extended, 
                                                         torch.linalg.inv(self.kriging_matrix)), dim=1)
