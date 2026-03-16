import warnings
warnings.filterwarnings('ignore')

import argparse
import time
import numpy as np
try:
    from ucimlrepo import fetch_ucirepo
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'ucimlrepo'. Install it with: pip install ucimlrepo"
    ) from exc

try:
    import torch
except ImportError:
    torch = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterative SVD noise scoring with optional GPU acceleration."
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Execution backend: auto, cpu, or gpu (default: auto).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of top noisy instances to display (default: 10).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="GPU candidate batch size for evaluation (default: 256).",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="CUDA GPU index to use when backend is gpu/auto (default: 0).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print progress/timing every N outer loops (default: 25).",
    )
    parser.add_argument(
        "--svd-every",
        type=int,
        default=1,
        help="Print SVD before/after/delta every N outer loops (default: 1).",
    )
    return parser.parse_args()


def compute_iterative_scores_numpy(X_np, log_every, svd_every):
    num_instances = X_np.shape[0]
    index_removed_at_step = np.zeros(num_instances, dtype=int)
    noise_score_at_step = np.zeros(num_instances, dtype=np.float64)

    X_running = np.copy(X_np)
    singular_values_shape = None
    t_all_start = time.perf_counter()

    for i in range(num_instances):
        t_loop_start = time.perf_counter()
        _, s_running, _ = np.linalg.svd(X_running, full_matrices=False)
        if singular_values_shape is None:
            singular_values_shape = s_running.shape

        all_scores = np.full(num_instances, -np.inf, dtype=np.float64)
        comp_time_sum = 0.0
        comp_count = 0

        for j in range(num_instances):
            if np.all(X_running[j, :] == 0):
                continue

            t_comp_start = time.perf_counter()
            X_reduced = np.copy(X_running)
            X_reduced[j, :] = 0
            _, s_new, _ = np.linalg.svd(X_reduced, full_matrices=False)

            min_len = min(len(s_running), len(s_new))
            all_scores[j] = np.linalg.norm(s_new[:min_len] - s_running[:min_len])
            comp_time_sum += (time.perf_counter() - t_comp_start)
            comp_count += 1

        val = np.max(all_scores)
        idx_to_zero = int(np.argmax(all_scores))

        index_removed_at_step[i] = idx_to_zero
        noise_score_at_step[i] = val
        X_after = np.copy(X_running)
        X_after[idx_to_zero, :] = 0
        _, s_after, _ = np.linalg.svd(X_after, full_matrices=False)
        min_len_delta = min(len(s_running), len(s_after))
        s_delta = s_after[:min_len_delta] - s_running[:min_len_delta]
        X_running[idx_to_zero, :] = 0

        t_loop = time.perf_counter() - t_loop_start
        avg_comp_time = (comp_time_sum / comp_count) if comp_count else 0.0
        if (i + 1) % log_every == 0 or i < 3 or i == num_instances - 1:
            print(
                f"  Processed {i + 1}/{num_instances} | "
                f"loop_time={t_loop:.3f}s | "
                f"avg_comp_time={avg_comp_time:.6f}s | "
                f"computations={comp_count}"
            )
        if (i + 1) % svd_every == 0 or i < 3 or i == num_instances - 1:
            print(f"    Loop {i + 1}: selected_index={idx_to_zero}, score={val:.6f}")
            print(f"    SVD before: {np.array2string(s_running, precision=4, separator=', ')}")
            print(f"    SVD after : {np.array2string(s_after, precision=4, separator=', ')}")
            print(f"    SVD delta : {np.array2string(s_delta, precision=4, separator=', ')}")

    total_time = time.perf_counter() - t_all_start
    print(f"Total scoring time: {total_time:.3f}s")

    return noise_score_at_step, index_removed_at_step, singular_values_shape


def compute_iterative_scores_torch(X_np, chunk_size, device, log_every, svd_every):
    num_instances = X_np.shape[0]
    index_removed_at_step = np.zeros(num_instances, dtype=int)
    noise_score_at_step = np.zeros(num_instances, dtype=np.float64)

    X_running = torch.tensor(X_np, dtype=torch.float32, device=device)
    singular_values_shape = None
    t_all_start = time.perf_counter()

    for i in range(num_instances):
        t_loop_start = time.perf_counter()
        s_running = torch.linalg.svdvals(X_running)
        if singular_values_shape is None:
            singular_values_shape = tuple(s_running.shape)

        row_nonzero_mask = torch.any(X_running != 0, dim=1)
        candidate_indices = torch.nonzero(row_nonzero_mask, as_tuple=False).squeeze(1)

        if candidate_indices.numel() == 0:
            break

        best_score = -float("inf")
        best_idx = int(candidate_indices[0].item())
        best_s_after = None
        comp_time_sum = 0.0
        comp_count = 0

        for start in range(0, candidate_indices.numel(), chunk_size):
            t_comp_start = time.perf_counter()
            chunk = candidate_indices[start:start + chunk_size]
            batch_size = int(chunk.shape[0])

            X_batch = X_running.unsqueeze(0).repeat(batch_size, 1, 1)
            X_batch[torch.arange(batch_size, device=device), chunk, :] = 0

            s_new_batch = torch.linalg.svdvals(X_batch)
            min_len = min(s_running.shape[-1], s_new_batch.shape[-1])
            diff = s_new_batch[:, :min_len] - s_running[:min_len].unsqueeze(0)
            scores = torch.linalg.norm(diff, dim=1)

            chunk_best_score, chunk_best_pos = torch.max(scores, dim=0)
            chunk_best_score_val = float(chunk_best_score.item())
            if chunk_best_score_val > best_score:
                best_score = chunk_best_score_val
                best_idx = int(chunk[int(chunk_best_pos.item())].item())
                best_s_after = s_new_batch[int(chunk_best_pos.item()), :].clone()

            comp_time_sum += (time.perf_counter() - t_comp_start)
            comp_count += batch_size

        index_removed_at_step[i] = best_idx
        noise_score_at_step[i] = best_score
        s_running_np = s_running.detach().cpu().numpy()
        if best_s_after is None:
            s_after_np = s_running_np.copy()
        else:
            s_after_np = best_s_after.detach().cpu().numpy()
        min_len_delta = min(len(s_running_np), len(s_after_np))
        s_delta_np = s_after_np[:min_len_delta] - s_running_np[:min_len_delta]
        X_running[best_idx, :] = 0

        t_loop = time.perf_counter() - t_loop_start
        avg_comp_time = (comp_time_sum / comp_count) if comp_count else 0.0
        if (i + 1) % log_every == 0 or i < 3 or i == num_instances - 1:
            print(
                f"  Processed {i + 1}/{num_instances} | "
                f"loop_time={t_loop:.3f}s | "
                f"avg_comp_time={avg_comp_time:.6f}s | "
                f"computations={comp_count}"
            )
        if (i + 1) % svd_every == 0 or i < 3 or i == num_instances - 1:
            print(f"    Loop {i + 1}: selected_index={best_idx}, score={best_score:.6f}")
            # print(f"    SVD before: {np.array2string(s_running_np, precision=4, separator=', ')}")
            # print(f"    SVD after : {np.array2string(s_after_np, precision=4, separator=', ')}")
            # print(f"    SVD delta : {np.array2string(s_delta_np, precision=4, separator=', ')}")

    total_time = time.perf_counter() - t_all_start
    print(f"Total scoring time: {total_time:.3f}s")

    return noise_score_at_step, index_removed_at_step, singular_values_shape


def compute_noise_scores(X_np, backend, chunk_size, gpu_index, log_every, svd_every):
    if backend == "cpu":
        return (
            compute_iterative_scores_numpy(X_np, log_every=log_every, svd_every=svd_every),
            "cpu/numpy",
        )

    if backend in {"auto", "gpu"}:
        if torch is not None and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if gpu_index < 0 or gpu_index >= device_count:
                raise RuntimeError(
                    f"Invalid --gpu-index {gpu_index}. Available CUDA device indices: 0 to {device_count - 1}."
                )

            device = torch.device(f"cuda:{gpu_index}")
            return (
                compute_iterative_scores_torch(
                    X_np,
                    chunk_size=chunk_size,
                    device=device,
                    log_every=log_every,
                    svd_every=svd_every,
                ),
                f"gpu/torch ({torch.cuda.get_device_name(gpu_index)})",
            )
        if backend == "gpu":
            raise RuntimeError(
                "GPU backend requested, but CUDA is not available in this Python environment. "
                "Install a CUDA-enabled PyTorch build or run with --backend cpu."
            )

    return (
        compute_iterative_scores_numpy(X_np, log_every=log_every, svd_every=svd_every),
        "cpu/numpy",
    )

args = parse_args()

# Fetch dataset
spambase = fetch_ucirepo(id=94)

# data (as pandas dataframes)
X = spambase.data.features
y = spambase.data.targets

# metadata
print("Dataset Metadata:")
print(spambase.metadata)
print("\n")

print(f"Total number of instances (rows) in features (X): {X.shape[0]}")
print(f"Total number of instances (rows) in target (y): {y.shape[0]}")

X_total_np = X.values.astype(np.float32)
num_instances = X_total_np.shape[0]

print("\nStarting iterative SVD noise scoring...")
(noise_scores, index_removed_at_step, singular_values_shape), backend_name = compute_noise_scores(
    X_total_np,
    backend=args.backend,
    chunk_size=max(1, args.chunk_size),
    gpu_index=args.gpu_index,
    log_every=max(1, args.log_every),
    svd_every=max(1, args.svd_every),
)

print(f"Backend: {backend_name}")
print(f"Initial singular values shape: {singular_values_shape}")
print("Iterative SVD noise scoring complete.")

topk = max(1, min(args.topk, num_instances))
top_step_indices = np.arange(topk)

print(f"\nFirst {topk} removed instances (MATLAB-style ordered influence):")
for step_idx in top_step_indices:
    original_idx = index_removed_at_step[step_idx]
    print(f"Original Index: {original_idx}, Noise Score: {noise_scores[step_idx]:.6f}")