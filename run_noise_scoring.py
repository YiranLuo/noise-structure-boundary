#!/usr/bin/env python3
"""
Noise Scoring Pipeline - Standalone script with timestamped logging and checkpoints.
Run this in a tmux session: tmux new-session -d -s scoring 'python run_noise_scoring.py'
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import json
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")

ALPHA = 0.1
MAXITERS = 300
LAMBDAS = {'aggressive': 0.1, 'stealthy': 50.0}
REQUESTED_N_SAMPLES = 2000
SCORING_N = None  # Will be set dynamically
TOPK = 10
CHUNK_SIZE = 256

log(f"Hyperparameters: ALPHA={ALPHA}, MAXITERS={MAXITERS}, REQUESTED_N_SAMPLES={REQUESTED_N_SAMPLES}")

# ============================================================================
# DATA LOADING
# ============================================================================
log("Fetching Spambase dataset...")
try:
    spambase = fetch_ucirepo(id=94)
    X_raw = spambase.data.features
    y_raw = spambase.data.targets
    df = pd.concat([X_raw, y_raw], axis=1)
    target_col = y_raw.columns[0]
    feature_names = X_raw.columns.tolist()
    log(f"✓ Dataset loaded: {df.shape[0]} rows, {len(feature_names)} features")
except Exception as e:
    log(f"✗ Failed to load dataset: {e}")
    exit(1)

# Preprocessing
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
scaler = MinMaxScaler()
df[feature_names] = scaler.fit_transform(df[feature_names])
weights = abs(df.corr()[target_col]).drop(target_col).values
weights = weights / (np.linalg.norm(weights) + 1e-8)
bounds = [df[feature_names].min().values, df[feature_names].max().values]
log(f"✓ Dataset preprocessed | Class dist: {dict(df[target_col].value_counts())}")

# ============================================================================
# MODEL DEFINITION & TRAINING
# ============================================================================
class SpambaseNet(nn.Module):
    def __init__(self, D_in):
        super(SpambaseNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(D_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2), nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.layer(x).squeeze(0)
        return self.layer(x)

log("Training neural network...")
X_tensor = torch.FloatTensor(df[feature_names].values)
y_tensor = torch.nn.functional.one_hot(torch.LongTensor(df[target_col].values.astype(int))).float()
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

model = SpambaseNet(len(feature_names)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        log(f"  Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    train_acc = (model(X_train).argmax(dim=1) == y_train.argmax(dim=1)).float().mean()
    test_acc = (model(X_test).argmax(dim=1) == y_test.argmax(dim=1)).float().mean()
log(f"✓ Training complete | Train: {train_acc:.2%}, Test: {test_acc:.2%}")

# ============================================================================
# ADVERSARIAL ATTACK GENERATION
# ============================================================================
def clip(current, low_bound, up_bound, device):
    low_bound = torch.FloatTensor(low_bound).to(device)
    up_bound = torch.FloatTensor(up_bound).to(device)
    return torch.max(torch.min(current, up_bound), low_bound)

def lowProFool_gpu(x, model, weights, bounds, maxiters, alpha, lambda_, device):
    x = x.to(device)
    v = torch.FloatTensor(np.array(weights)).to(device)
    r = torch.FloatTensor(1e-4 * np.ones(x.shape)).to(device)
    r.requires_grad = True
    
    with torch.no_grad():
        output = model(x)
        orig_pred = output.argmax().cpu().item()
    
    target_pred = 1 - orig_pred
    target = torch.tensor([0., 1.] if target_pred == 1 else [1., 0.]).to(device)
    bce = nn.BCELoss()
    
    for _ in range(maxiters):
        if r.grad is not None:
            r.grad.zero_()
        output = model(x + r)
        loss_bce = bce(output, target)
        loss_l2 = torch.sqrt(torch.sum((v * r) ** 2))
        loss = loss_bce + lambda_ * loss_l2
        loss.backward(retain_graph=True)
        with torch.no_grad():
            r_new = r - alpha * r.grad
        r = r_new.clone().detach().requires_grad_(True)
    
    x_adv = clip(x + r, bounds[0], bounds[1], device)
    with torch.no_grad():
        adv_pred = model(x_adv).argmax().cpu().item()
    return orig_pred, adv_pred, x_adv.detach().cpu().numpy()

actual_n_samples = min(REQUESTED_N_SAMPLES, X_test.shape[0])
log(f"Generating adversarial examples (requested={REQUESTED_N_SAMPLES}, actual={actual_n_samples})...")
test_samples = X_test[:actual_n_samples].cpu()
original_data = test_samples.numpy()
adversarial_data, attack_success = {}, {}

for attack_type, lambda_val in LAMBDAS.items():
    log(f"  LowProFool ({attack_type}, λ={lambda_val})...")
    adv_samples, successful = [], 0
    for i, x in enumerate(test_samples):
        orig_pred, adv_pred, x_adv = lowProFool_gpu(x, model, weights, bounds, MAXITERS, ALPHA, lambda_val, device)
        adv_samples.append(x_adv)
        if orig_pred != adv_pred:
            successful += 1
        if (i + 1) % 500 == 0:
            log(f"    Processed {i+1}/{actual_n_samples}")
    
    adversarial_data[attack_type] = np.array(adv_samples)
    attack_success[attack_type] = successful / actual_n_samples
    perturbation = adversarial_data[attack_type] - original_data
    avg_l2_norm = np.mean(np.linalg.norm(perturbation, axis=1))
    log(f"  ✓ {attack_type}: Success {attack_success[attack_type]:.2%}, Avg L2: {avg_l2_norm:.6f}")

# ============================================================================
# RANDOM NOISE BASELINE
# ============================================================================
log("Creating random noise baseline...")
aggressive_perturbation = adversarial_data['aggressive'] - original_data
target_l2_norms = np.linalg.norm(aggressive_perturbation, axis=1)
np.random.seed(42)
random_noise = np.random.randn(*original_data.shape)
noise_norms = np.linalg.norm(random_noise, axis=1, keepdims=True)
random_noise = random_noise / (noise_norms + 1e-8) * target_l2_norms.reshape(-1, 1)
noisy_data = np.clip(original_data + random_noise, bounds[0], bounds[1])
log(f"✓ Random baseline created | Avg L2: {np.mean(np.linalg.norm(noisy_data - original_data, axis=1)):.6f}")

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        comp_count = 0

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
            comp_count += batch_size

        index_removed_s1[i] = best_idx
        noise_score1_s1[i] = best_score1
        X_running[best_idx, :] = 0

        if (i + 1) % log_every == 0 or i < 3 or i == num_instances - 1:
            t_loop = time.perf_counter() - t_loop_start
            log(f"    Iter {i + 1}/{num_instances} | time={t_loop:.3f}s | score1={best_score1:.6f}")

    total_time = time.perf_counter() - t_all_start
    log(f"  Total scoring time: {total_time:.3f}s")
    return noise_score1_s1, index_removed_s1, noise_score2_all, singular_values_shape

SCORING_N = original_data.shape[0]
scenario_inputs = {
    'aggressive': _to_2d_float32(adversarial_data['aggressive'][:SCORING_N]),
    'stealthy': _to_2d_float32(adversarial_data['stealthy'][:SCORING_N]),
    'random_noise_baseline': _to_2d_float32(noisy_data[:SCORING_N]),
}

svd_results = {}
scoring_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"SVD device: {scoring_device}")

for name, X_scenario in scenario_inputs.items():
    log(f"\n[{name}] Starting noise scoring | shape={X_scenario.shape}")
    noise_scores_1, index_removed_s1, noise_scores_2, singular_values_shape = compute_iterative_scores_torch(
        X_scenario, chunk_size=CHUNK_SIZE, device=scoring_device, log_every=50
    )

    topk = min(TOPK, len(noise_scores_1), len(noise_scores_2))
    log(f"[{name}] Top {topk} by Score-1:")
    for rank in range(topk):
        idx = int(index_removed_s1[rank])
        score1 = float(noise_scores_1[rank])
        score2_of_idx = float(noise_scores_2[idx])
        log(f"  {rank + 1:2d}. Idx={idx:4d}, S1={score1:.6f}, S2={score2_of_idx:.6f}")

    svd_results[name] = {
        'backend': f'torch/{scoring_device}',
        'shape': X_scenario.shape,
        'singular_values_shape': singular_values_shape,
        'score1': {
            'noise_scores': noise_scores_1,
            'index_removed_at_step': index_removed_s1,
        },
        'score2': {
            'noise_scores': noise_scores_2,
        },
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

log(f"\n=== Pipeline Complete ===")
log(f"Results directory: {output_dir}")
log(f"All outputs timestamped: {TIMESTAMP}")
