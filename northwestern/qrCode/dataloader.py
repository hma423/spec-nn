import os
import glob
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class DiplexDataset(Dataset):
    """
    Deterministic versioned dataset.

    For a dataset with num_raw raw files:
      augment_factor = 1:
          idx = i              -> original

      augment_factor = 4:
          idx = i + 0*num_raw  -> original
          idx = i + 1*num_raw  -> flip_lr
          idx = i + 2*num_raw  -> flip_ud
          idx = i + 3*num_raw  -> rot180
    """

    def __init__(
        self,
        input_files: List[str],
        output_dir: str,
        n: int = 5,
        transform=None,
        augment_factor: int = 1,
    ):
        self.n = n
        self.transform = transform
        self.max_resolution = (1, 80, 80)

        if augment_factor not in (1, 4):
            raise ValueError(f"augment_factor must be 1 or 4, got {augment_factor}")
        self.augment_factor = augment_factor

        self.input_files = list(input_files)
        self.output_dir = output_dir
        self.output_files = [
            os.path.join(self.output_dir, os.path.basename(f)) for f in self.input_files
        ]

        # Keep exactly these 10 S-parameters:
        # 11, 12, 13, 14, 22, 23, 24, 33, 34, 44
        #
        # CSV columns:
        # 0  Frequency (GHz)
        # 1  MAG[S11], 2  ANG[S11]
        # 3  MAG[S12], 4  ANG[S12]
        # 5  MAG[S13], 6  ANG[S13]
        # 7  MAG[S14], 8  ANG[S14]
        # 9  MAG[S21], 10 ANG[S21]
        # 11 MAG[S22], 12 ANG[S22]
        # 13 MAG[S23], 14 ANG[S23]
        # 15 MAG[S24], 16 ANG[S24]
        # 17 MAG[S31], 18 ANG[S31]
        # 19 MAG[S32], 20 ANG[S32]
        # 21 MAG[S33], 22 ANG[S33]
        # 23 MAG[S34], 24 ANG[S34]
        # 25 MAG[S41], 26 ANG[S41]
        # 27 MAG[S42], 28 ANG[S42]
        # 29 MAG[S43], 30 ANG[S43]
        # 31 MAG[S44], 32 ANG[S44]
        self.keep_indices = [
            (1, 2),    # S11
            (3, 4),    # S12
            (5, 6),    # S13
            (7, 8),    # S14
            (11, 12),  # S22
            (13, 14),  # S23
            (15, 16),  # S24
            (21, 22),  # S33
            (23, 24),  # S34
            (31, 32),  # S44
        ]
        self.cols = [i for pair in self.keep_indices for i in pair]
        self.s_pairs = ["11", "12", "13", "14", "22", "23", "24", "33", "34", "44"]

        self.num_raw = len(self.input_files)
        if self.num_raw == 0:
            raise ValueError("DiplexDataset received 0 input files.")

    def __len__(self):
        return self.num_raw * self.augment_factor

    def __getitem__(self, idx):
        base_idx = idx % self.num_raw
        transform_id = idx // self.num_raw  # 0,1,2,3 if augment_factor=4; only 0 if =1

        # -----------------------------
        # Input
        # -----------------------------
        input_path = self.input_files[base_idx]
        input_arr = np.loadtxt(input_path, delimiter=",").astype(np.float32)
        input_img = torch.from_numpy(input_arr)

        input_img = torch.nn.functional.interpolate(
            input_img.unsqueeze(0).unsqueeze(0),
            size=self.max_resolution[1:],
            mode="nearest",
        ).squeeze(0)  # [1, 80, 80]

        # -----------------------------
        # Output
        # -----------------------------
        output_path = self.output_files[base_idx]
        df = pd.read_csv(output_path, sep=",", skiprows=1)

        # Keep only numeric rows in first column
        df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()]

        freqs = df.iloc[:, 0].astype(float).to_numpy()
        data = df.iloc[:, self.cols].astype(float).to_numpy()

        num_intervals = 2 ** self.n
        edges = np.linspace(freqs.min(), freqs.max(), num_intervals + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        idxs = np.abs(freqs[:, None] - centers[None, :]).argmin(axis=0)

        chosen_rows = data[idxs]  # shape: (num_intervals, 2 * len(s_pairs))

        # Wrap angles to [-180, 180)
        angle_cols = np.arange(1, chosen_rows.shape[1], 2)
        angles = chosen_rows[:, angle_cols]
        chosen_rows[:, angle_cols] = ((angles + 180.0) % 360.0) - 180.0

        output_vec = torch.tensor(chosen_rows.flatten(), dtype=torch.float32)

        if self.transform is not None:
            input_img, output_vec = self.transform(
                input_img, output_vec, self.s_pairs, transform_id
            )

        return input_img, output_vec


class FixedPhysicalTransform:
    """
    Deterministic transform by transform_id:
      0 -> original
      1 -> flip_lr
      2 -> flip_ud
      3 -> rot180

    Assumptions:
      - reciprocal network: S_ij = S_ji
      - port layout:
            1  2
            3  4
    """

    def __call__(self, x, y, s_pairs, transform_id):
        if transform_id == 0:
            # original
            swap = {
                "11": "11",
                "12": "12",
                "13": "13",
                "14": "14",
                "22": "22",
                "23": "23",
                "24": "24",
                "33": "33",
                "34": "34",
                "44": "44",
            }

        elif transform_id == 1:
            # left-right flip: 1<->2, 3<->4
            x = torch.flip(x, dims=[2])
            swap = {
                "11": "22",
                "12": "12",
                "13": "24",
                "14": "23",
                "22": "11",
                "23": "14",
                "24": "13",
                "33": "44",
                "34": "34",
                "44": "33",
            }

        elif transform_id == 2:
            # up-down flip: 1<->3, 2<->4
            x = torch.flip(x, dims=[1])
            swap = {
                "11": "33",
                "12": "34",
                "13": "13",
                "14": "23",
                "22": "44",
                "23": "14",
                "24": "24",
                "33": "11",
                "34": "12",
                "44": "22",
            }

        elif transform_id == 3:
            # rotate 180: 1<->4, 2<->3
            x = torch.flip(x, dims=[1, 2])
            swap = {
                "11": "44",
                "12": "34",
                "13": "24",
                "14": "14",
                "22": "33",
                "23": "23",
                "24": "13",
                "33": "22",
                "34": "12",
                "44": "11",
            }

        else:
            raise ValueError(f"Unsupported transform_id: {transform_id}")

        s_map = [s_pairs.index(swap[s]) for s in s_pairs]

        num_freq = y.shape[0] // (len(s_pairs) * 2)
        y = y.view(num_freq, len(s_pairs), 2)[:, s_map, :].reshape(-1)
        return x, y


def _get_all_input_files(data_root: str) -> Tuple[List[str], str]:
    input_dir = os.path.join(data_root, "geometries")
    output_dir = os.path.join(data_root, "son_files", "QR_S_Data")

    all_input_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if len(all_input_files) == 0:
        raise FileNotFoundError(f"No geometry CSVs found in: {input_dir}")

    return all_input_files, output_dir


def build_loaders_dual_val(
    data_root,
    n=5,
    batch_size=16,
    seed=42,
    num_workers=0,
    pin_memory=True,
    subset_fraction=1.0,
    valB_raw_count=100,   # strict raw-split validation size
    valA_aug_count=100,   # optimistic augmented-split validation size
):
    """
    Build loaders with:
      TRAIN  : from raw-train-pool -> augment x4 -> remove VAL_A samples
      VAL_A  : 100 samples drawn from the augmented train-pool family
      VAL_B  : 100 raw files, original only, raw-disjoint from train

    Example when total raw = 1000:
      raw total      = 1000
      raw VAL_B      = 100
      raw train pool = 900

      augmented train pool = 900 * 4 = 3600
      VAL_A               = 100
      TRAIN               = 3500
    """
    if not (0 < subset_fraction <= 1.0):
        raise ValueError(f"subset_fraction must be in (0,1], got {subset_fraction}")

    all_input_files, output_dir = _get_all_input_files(data_root)
    num_total_raw = len(all_input_files)

    if valB_raw_count < 1 or valB_raw_count >= num_total_raw:
        raise ValueError(
            f"valB_raw_count must be in [1, {num_total_raw - 1}], got {valB_raw_count}"
        )

    # ------------------------------------------------------------
    # Step 1: split raw files into train-pool raw and VAL_B raw
    # ------------------------------------------------------------
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_total_raw)

    valB_idx = perm[:valB_raw_count]
    train_pool_idx = perm[valB_raw_count:]

    raw_valB_files = [all_input_files[i] for i in valB_idx]
    raw_train_pool_files = [all_input_files[i] for i in train_pool_idx]

    # Optional nested subset on raw train pool
    num_train_pool_raw = len(raw_train_pool_files)
    num_keep_raw = max(1, int(math.floor(num_train_pool_raw * subset_fraction)))

    subset_rng = np.random.default_rng(seed)
    subset_perm = subset_rng.permutation(num_train_pool_raw)
    raw_train_subset_files = [raw_train_pool_files[i] for i in subset_perm[:num_keep_raw]]

    # ------------------------------------------------------------
    # Step 2: build augmented dataset from raw train subset
    # ------------------------------------------------------------
    full_train_aug_ds = DiplexDataset(
        input_files=raw_train_subset_files,
        output_dir=output_dir,
        n=n,
        transform=FixedPhysicalTransform(),
        augment_factor=4,
    )

    total_aug_train = len(full_train_aug_ds)
    if valA_aug_count < 1 or valA_aug_count >= total_aug_train:
        raise ValueError(
            f"valA_aug_count must be in [1, {total_aug_train - 1}], got {valA_aug_count}"
        )

    # ------------------------------------------------------------
    # Step 3: split augmented train-pool into TRAIN and VAL_A
    # ------------------------------------------------------------
    train_count = total_aug_train - valA_aug_count
    split_generator = torch.Generator().manual_seed(seed)

    train_ds, valA_ds = random_split(
        full_train_aug_ds,
        [train_count, valA_aug_count],
        generator=split_generator,
    )

    # ------------------------------------------------------------
    # Step 4: build VAL_B (raw only, no augmentation)
    # ------------------------------------------------------------
    valB_ds = DiplexDataset(
        input_files=raw_valB_files,
        output_dir=output_dir,
        n=n,
        transform=None,
        augment_factor=1,
    )

    # ------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valA_loader = DataLoader(
        valA_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valB_loader = DataLoader(
        valB_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    meta = {
        "all_input_files": all_input_files,
        "raw_train_pool_files": raw_train_pool_files,
        "raw_train_subset_files": raw_train_subset_files,
        "raw_valB_files": raw_valB_files,
        "full_train_aug_ds": full_train_aug_ds,
        "train_ds": train_ds,
        "valA_ds": valA_ds,
        "valB_ds": valB_ds,
        "subset_fraction": subset_fraction,
        "valB_raw_count": valB_raw_count,
        "valA_aug_count": valA_aug_count,
        "num_total_raw": num_total_raw,
        "num_train_pool_raw": len(raw_train_pool_files),
        "num_train_subset_raw": len(raw_train_subset_files),
        "num_aug_train_total": total_aug_train,
        "num_train_final": len(train_ds),
        "num_valA_final": len(valA_ds),
        "num_valB_final": len(valB_ds),
    }

    print("[INFO] Dual-validation setup (requested version)")
    print(f"  Raw total files:         {num_total_raw}")
    print(f"  Raw VAL_B files:         {len(raw_valB_files)}")
    print(f"  Raw train pool files:    {len(raw_train_pool_files)}")
    print(f"  Raw train subset files:  {len(raw_train_subset_files)}")
    print(f"  Augmented train pool:    {len(full_train_aug_ds)} (= {len(raw_train_subset_files)} x 4)")
    print(f"  TRAIN size:              {len(train_ds)}")
    print(f"  VAL_A size:              {len(valA_ds)}")
    print(f"  VAL_B size:              {len(valB_ds)}")

    return train_loader, valA_loader, valB_loader, meta
