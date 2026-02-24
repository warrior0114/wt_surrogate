from __future__ import annotations
import os
from typing import List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from wt_surrogate.features.otf import OTFFeatureExtractor, OTFFeatureConfig


class NpzCaseDataset(Dataset):
    """Simple Dataset that yields (wind_features, inputs, outputs) for each NPZ file.

    - Uses precomputed 'features' if present.
    - Otherwise extracts features on-the-fly using OTFFeatureExtractor.
    """
    def __init__(
        self,
        data_dir: str,
        files: Optional[Sequence[str]] = None,
        input_param_names: Optional[List[str]] = None,
        output_param_names: Optional[List[str]] = None,
        kept_feature_indices: Optional[List[int]] = None,
        weights_dir: Optional[str] = None,
        device: str = "cpu",
        batch: int = 2048,
        bts_root: Optional[str] = None,
    ):
        self.data_dir = data_dir
        if files is None:
            self.files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.npz')])
        else:
            self.files = [os.path.basename(f) for f in files]
        if not self.files:
            raise FileNotFoundError(f"No .npz files found under: {data_dir}")

        self.input_param_names = input_param_names
        self.output_param_names = output_param_names
        self.kept_feature_indices = kept_feature_indices

        self._otf = None
        if weights_dir is not None:
            self._otf = OTFFeatureExtractor(OTFFeatureConfig(weights_dir=weights_dir, device=device, batch=batch, bts_root=bts_root))

        # Use first file to build name indices (assume consistency)
        first = np.load(os.path.join(data_dir, self.files[0]), allow_pickle=True)
        self.input_names = [str(x) for x in (first['input_names'].tolist() if 'input_names' in first.files else [])]
        self.output_names = [str(x) for x in (first['output_names'].tolist() if 'output_names' in first.files else [])]
        self.feature_names = [str(x) for x in (first['feature_names'].tolist() if 'feature_names' in first.files else [])]

        self._in_map = {n: i for i, n in enumerate(self.input_names)}
        self._out_map = {n: i for i, n in enumerate(self.output_names)}

        self._in_idx = None
        if input_param_names is not None:
            self._in_idx = [self._in_map[n] for n in input_param_names]
        self._out_idx = None
        if output_param_names is not None:
            self._out_idx = [self._out_map[n] for n in output_param_names]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = os.path.join(self.data_dir, self.files[idx])
        d = np.load(path, allow_pickle=True)

        inputs = d['input_params'].astype(np.float32)
        outputs = d['output_params'].astype(np.float32)

        if 'features' in d.files:
            feats = d['features'].astype(np.float32)
        else:
            if self._otf is None:
                raise ValueError("NPZ has no 'features' and weights_dir was not provided.")
            feats, _ = self._otf.extract(d, row_idx=None)

        if self.kept_feature_indices is not None:
            feats = feats[:, self.kept_feature_indices]
        if self._in_idx is not None:
            inputs = inputs[:, self._in_idx]
        if self._out_idx is not None:
            outputs = outputs[:, self._out_idx]

        return torch.from_numpy(feats), torch.from_numpy(inputs), torch.from_numpy(outputs)
