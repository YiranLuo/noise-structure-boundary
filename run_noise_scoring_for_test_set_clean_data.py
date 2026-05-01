#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import time
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# LOGGING SETUP WITH TIMESTAMPS
# ============================================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = Path("logs") / TIMESTAMP
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "noise_scoring.log"

def log(msg):
    """Log message with timestamp to both stdout and file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{ts}] {msg}"
    print(full_msg)
    with open(LOG_FILE, 'a') as f:
        f.write(full_msg + '\n')

log(f"=== Noise Scoring Pipeline Started (Session: {TIMESTAMP}) ===")
log(f"Logs: {LOG_FILE}")

# ============================================================================
# DEVICE & HYPERPARAMETERS
# ============================================================================
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
log(f"Device: {DEVICE}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")

CHUNK_SIZE = 256
TOPK = 20
DATA_FILE = Path("experiments/data/spambase_50_50") / "train_test_data.npz"


def _load_test_set(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find dataset file: {npz_path}")

    data = np.load(npz_path)
    keys = list(data.keys())
    key_to_lower = {k: k.lower() for k in keys}

    def _pick_key(candidates):
        for key in keys:
            k = key_to_lower[key]
            if any(candidate in k for candidate in candidates):
                return key
        return None

    test_x_key = _pick_key(["test_x", "x_test", "test_data", "test_feature", "test_features"])
    test_y_key = _pick_key(["test_y", "y_test", "test_label", "test_labels", "test_target", "test_targets"])

    if test_x_key is None:
        raise KeyError(
            f"Could not locate test features in {npz_path}. "
            f"Available keys: {keys}"
        )

    X_test = np.asarray(data[test_x_key])
    y_test = np.asarray(data[test_y_key]) if test_y_key is not None else None

    log(f"Loaded dataset: {npz_path}")
    log(f"Available keys: {keys}")
    log(f"Using test features key: {test_x_key} | shape={X_test.shape}")
    if y_test is not None:
        log(f"Using test labels key: {test_y_key} | shape={y_test.shape}")
    else:
        log("No test labels key detected; continuing with features only.")

    return X_test, y_test, keys, test_x_key, test_y_key


# ============================================================================
# NOISE SCORING
# ============================================================================
def _to_2d_float32(arr):
    x = np.asarray(arr)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    return x.astype(np.float32)

def compute_iterative_scores_torch(X_np, chunk_size=256, device=None, log_every=50):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_instances = X_np.shape[0]
    t_all_start = time.perf_counter()

    log("  Computing Score-2 from initial full-matrix SVD...")
    X_full = torch.tensor(X_np, dtype=torch.float32, device=device)
    U_full, s_full, _ = torch.linalg.svd(X_full, full_matrices=False)
    singular_values_shape = tuple(s_full.shape)
    
    noise_score2_all = np.zeros(num_instances, dtype=np.float64)
    for row_idx in range(num_instances):
        u_row = U_full[row_idx, :]
        if u_row.shape[0] > 1:
            score2_val = float(torch.max(torch.abs(torch.diff(u_row))).item())
        else:
            score2_val = 0.0
        noise_score2_all[row_idx] = score2_val
    
    log("  Computing Score-1 by iterative greedy removal...")
    index_removed_s1 = np.zeros(num_instances, dtype=int)
    noise_score1_s1 = np.zeros(num_instances, dtype=np.float64)
    X_running = torch.tensor(X_np, dtype=torch.float32, device=device)

    for i in range(num_instances):
        t_loop_start = time.perf_counter()
        _, s_running, _ = torch.linalg.svd(X_running, full_matrices=False)
        row_nonzero_mask = torch.any(X_running != 0, dim=1)
        candidate_indices = torch.nonzero(row_nonzero_mask, as_tuple=False).squeeze(1)

        if candidate_indices.numel() == 0:
            break

        best_score1 = -float('inf')
        best_idx = int(candidate_indices[0].item())

        for start in range(0, candidate_indices.numel(), chunk_size):
            chunk = candidate_indices[start:start + chunk_size]
            batch_size = int(chunk.shape[0])
            X_batch = X_running.unsqueeze(0).repeat(batch_size, 1, 1)
            X_batch[torch.arange(batch_size, device=device), chunk, :] = 0
            s_new_batch = torch.linalg.svdvals(X_batch)
            min_len = min(s_running.shape[-1], s_new_batch.shape[-1])
            diff_s = s_new_batch[:, :min_len] - s_running[:min_len].unsqueeze(0)
            score1_batch = torch.linalg.norm(diff_s, dim=1)
            chunk_best_score1, chunk_best_pos = torch.max(score1_batch, dim=0)
            chunk_best_pos_int = int(chunk_best_pos.item())
            chunk_best_score1_val = float(chunk_best_score1.item())

            if chunk_best_score1_val > best_score1:
                best_score1 = chunk_best_score1_val
                best_idx = int(chunk[chunk_best_pos_int].item())

        index_removed_s1[i] = best_idx
        noise_score1_s1[i] = best_score1
        X_running[best_idx, :] = 0

        if (i + 1) % log_every == 0 or i < 3 or i == num_instances - 1:
            t_loop = time.perf_counter() - t_loop_start
            log(f"    Iter {i + 1}/{num_instances} | time={t_loop:.3f}s | score1={best_score1:.6f}")

    total_time = time.perf_counter() - t_all_start
    log(f"  Total scoring time: {total_time:.3f}s")
    return noise_score1_s1, index_removed_s1, noise_score2_all, singular_values_shape

X_test_raw, y_test, npz_keys, test_x_key, test_y_key = _load_test_set(DATA_FILE)
X_test = _to_2d_float32(X_test_raw)

name = 'test_clean_data'
log(f"\n[{name}] Starting noise scoring | shape={X_test.shape}")
noise_scores_1, index_removed_s1, noise_scores_2, singular_values_shape = compute_iterative_scores_torch(
    X_test, chunk_size=CHUNK_SIZE, device=DEVICE, log_every=50
)

topk = min(TOPK, len(noise_scores_1), len(noise_scores_2))
log(f"[{name}] Top {topk} by Score-1:")
for rank in range(topk):
    idx = int(index_removed_s1[rank])
    score1 = float(noise_scores_1[rank])
    score2_of_idx = float(noise_scores_2[idx])
    log(f"  {rank + 1:2d}. Idx={idx:4d}, S1={score1:.6f}, S2={score2_of_idx:.6f}")

svd_results = {
    name: {
        'backend': f'torch/{DEVICE}',
        'shape': X_test.shape,
        'singular_values_shape': singular_values_shape,
        'source': {
            'data_file': str(DATA_FILE),
            'npz_keys': npz_keys,
            'test_features_key': test_x_key,
            'test_labels_key': test_y_key,
            'num_test_samples': int(X_test.shape[0]),
            'num_features': int(X_test.shape[1]) if X_test.ndim > 1 else 1,
            'has_labels': y_test is not None,
        },
        'score1': {
            'noise_scores': noise_scores_1,
            'index_removed_at_step': index_removed_s1,
        },
        'score2': {
            'noise_scores': noise_scores_2,
        },
    }
}

# ============================================================================
# SAVE RESULTS WITH TIMESTAMPS
# ============================================================================
output_dir = Path("experiments/outputs/noise_scoring") / TIMESTAMP
output_dir.mkdir(parents=True, exist_ok=True)
log(f"\nSaving results to: {output_dir}")

def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    return obj

# JSON
json_path = output_dir / "svd_results.json"
with json_path.open("w") as f:
    json.dump(_json_safe(svd_results), f, indent=2)
log(f"✓ Saved JSON: {json_path}")

# CSVs
for scenario, data in svd_results.items():
    score1_data = data['score1']
    df_score1 = pd.DataFrame({
        "rank": np.arange(1, len(score1_data['noise_scores']) + 1),
        "removed_index": score1_data['index_removed_at_step'],
        "noise_score": score1_data['noise_scores'],
    })
    csv_path_s1 = output_dir / f"{scenario}_score1_ranking.csv"
    df_score1.to_csv(csv_path_s1, index=False)
    log(f"✓ Saved CSV: {csv_path_s1}")
    npy_path_s1_scores = output_dir / f"{scenario}_score1_noise_scores.npy"
    npy_path_s1_idx = output_dir / f"{scenario}_score1_removed_index.npy"
    np.save(npy_path_s1_scores, np.asarray(score1_data['noise_scores']))
    np.save(npy_path_s1_idx, np.asarray(score1_data['index_removed_at_step']))
    log(f"✓ Saved NPY: {npy_path_s1_scores}")
    log(f"✓ Saved NPY: {npy_path_s1_idx}")

    score2_data = data['score2']
    score2_all = score2_data['noise_scores']
    score2_ranking = np.argsort(score2_all)[::-1]
    df_score2 = pd.DataFrame({
        "rank": np.arange(1, len(score2_all) + 1),
        "sample_index": score2_ranking,
        "noise_score": score2_all[score2_ranking],
    })
    csv_path_s2 = output_dir / f"{scenario}_score2_ranking.csv"
    df_score2.to_csv(csv_path_s2, index=False)
    log(f"✓ Saved CSV: {csv_path_s2}")
    npy_path_s2_scores = output_dir / f"{scenario}_score2_noise_scores.npy"
    np.save(npy_path_s2_scores, np.asarray(score2_all))
    log(f"✓ Saved NPY: {npy_path_s2_scores}")

log(f"\n=== Pipeline Complete ===")
log(f"Results directory: {output_dir}")
log(f"All outputs timestamped: {TIMESTAMP}")
