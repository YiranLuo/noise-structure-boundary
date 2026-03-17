import warnings
warnings.filterwarnings('ignore')

import numpy as np
try:
    from ucimlrepo import fetch_ucirepo
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'ucimlrepo'. Install it with: pip install ucimlrepo"
    ) from exc

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
# The original notebook referenced 'df' here, but X is the features DataFrame.
# print(f"Total number of instances (rows) in original DataFrame (X): {X.shape[0]}")

# Perform initial SVD on X to get singular values for a quick check
U_initial, S_initial, Vh_initial = np.linalg.svd(X)
print(f"Initial singular values shape: {S_initial.shape}")

# SVD Noise Scoring Algorithm
# Convert X to a numpy array for SVD operations
X_total_np = X.values

# Initialize arrays to store results
num_instances = X_total_np.shape[0]
# index_removed will store the original index of the row that was zeroed out at each step.
index_removed_at_step = np.zeros(num_instances, dtype=int)
noise_score_at_step = np.zeros(num_instances)

X_running = np.copy(X_total_np)

print("\nStarting iterative SVD noise scoring...")

for i in range(num_instances):
    # Perform SVD on the current X_running
    # s_running contains the singular values (1D array)
    _, s_running, _ = np.linalg.svd(X_running, full_matrices=False)

    # Search for the point that maximized ||s_new - s_old||
    all_scores = np.zeros(num_instances) # Initialize scores for this iteration

    for j in range(num_instances):
        # Skip if this point has already been 'removed' (set to zeros)
        if np.all(X_running[j, :] == 0):
            all_scores[j] = -np.inf # Assign a very low score so it's not picked again
            continue

        X_reduced = np.copy(X_running)
        X_reduced[j, :] = 0  # Temporarily 'remove' the j-th point by zeroing its row

        # Perform SVD on the reduced matrix
        _, s_new, _ = np.linalg.svd(X_reduced, full_matrices=False)

        # Ensure singular value arrays have the same length for comparison
        min_len = min(len(s_running), len(s_new))

        # Compute the singular value difference score
        all_scores[j] = np.linalg.norm(s_new[:min_len] - s_running[:min_len])

    # Find the index of the point with the maximum score in the current X_running
    val = np.max(all_scores)
    idx_to_zero = np.argmax(all_scores)

    # Store the original index (which is `idx_to_zero` since we only zero out, not delete rows)
    # and the noise score for this step.
    index_removed_at_step[i] = idx_to_zero
    noise_score_at_step[i] = val

    # 'Remove' the point from X_running by setting its row to zeros
    X_running[idx_to_zero, :] = 0

    if (i + 1) % 100 == 0 or i == num_instances - 1:
        print(f"  Processed {i + 1}/{num_instances} instances.")

print("Iterative SVD noise scoring complete.")

# Prepare to display top noisy instances
# Create a list of (noise_score, original_index) tuples from the steps
noisy_points = []
for i in range(num_instances):
    noisy_points.append((noise_score_at_step[i], index_removed_at_step[i]))

# Sort by noise_score in descending order
noisy_points.sort(key=lambda x: x[0], reverse=True)

print("\nTop 10 'noisy' instances based on SVD impact (original index and score):")
for k in range(min(10, num_instances)):
    score, original_idx = noisy_points[k]
    print(f"Original Index: {original_idx}, Noise Score: {score:.4f}")