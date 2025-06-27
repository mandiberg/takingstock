import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm
import os
from scipy import stats
import csv

# Input CSV folder and output heatmap folder
CSV_FOLDER = "/Users/michaelmandiberg/Documents/GitHub/facemap/utilities/data/october_fusion_clusters"
HEATMAP_FOLDER = os.path.join(CSV_FOLDER, "heatmaps")
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

# Constants
POSE_COUNT = 32
IHG_COUNT = 128
LOG_ZERO = 10
CLIP_MAX = 10000  # Max value to cap heatmap color scale




# === Combined Heatmap Across All Topics ===
print("Generating combined heatmap...")

# Initialize combined matrix
combined_matrix = np.zeros((POSE_COUNT, IHG_COUNT))
combined_max = CLIP_MAX * 10
combined_zero = LOG_ZERO * 5
# Re-loop through all CSVs to aggregate
for topic_id in range(64):
    filename = f"topic_{topic_id}.csv"
    filepath = os.path.join(CSV_FOLDER, filename)

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['ihp_cluster'] = df['ihp_cluster'].astype(int)
        gesture_cols = [f'ihg_{i}' for i in range(IHG_COUNT)]
        clipped_df = df[gesture_cols].clip(upper=CLIP_MAX)

        for idx, row in df.iterrows():
            pose_id = int(row['ihp_cluster'])
            if 0 <= pose_id < POSE_COUNT:
                combined_matrix[pose_id, :] += clipped_df.loc[idx].values

# Avoid log(0) by setting zeros to a small value (e.g., 1)
log_matrix = combined_matrix.copy()
log_matrix[log_matrix == 0] = 1

# Use YlOrRd colormap with white for zero
cmap = plt.get_cmap("YlOrRd")
cmap_with_white = cmap(np.linspace(0, 1, 256))
cmap_with_white[0] = [1, 1, 1, 1]  # Set lowest value to white
custom_cmap = ListedColormap(cmap_with_white)

# Plot combined log heatmap
plt.figure(figsize=(16, 8))
ax = sns.heatmap(
    log_matrix,
    cmap=custom_cmap,
    norm=LogNorm(vmin=combined_zero, vmax=combined_max),
    cbar_kws={"label": "Count (log scale)"}
)
ax.set_title("Combined Heatmap (All Topics)", fontsize=16)
ax.set_xlabel("ihg_id")
ax.set_ylabel("pose_id")

# Save combined heatmap
output_path = os.path.join(HEATMAP_FOLDER, "combined_heatmap.png")
plt.tight_layout()
plt.savefig(output_path)
plt.close()

# Load all heatmaps into a 3D array: (topics, poses, gestures)
print("Loading data into 3D array for mean and mode calculations...")
all_heatmaps = []

for topic_id in range(64):
    filename = f"topic_{topic_id}.csv"
    filepath = os.path.join(CSV_FOLDER, filename)

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['ihp_cluster'] = df['ihp_cluster'].astype(int)
        gesture_cols = [f'ihg_{i}' for i in range(IHG_COUNT)]
        clipped_df = df[gesture_cols].clip(upper=CLIP_MAX)

        heatmap = np.zeros((POSE_COUNT, IHG_COUNT))

        for idx, row in df.iterrows():
            pose_id = int(row['ihp_cluster'])
            if 0 <= pose_id < POSE_COUNT:
                heatmap[pose_id, :] = clipped_df.loc[idx].values

        all_heatmaps.append(heatmap)

# Convert to 3D NumPy array: (topics, poses, gestures)
heatmap_stack = np.array(all_heatmaps)

# === Compute global min and max difference from mean ===
diff_range_min = np.inf
diff_range_max = -np.inf


# === MEAN HEATMAP ===
mean_matrix = np.mean(heatmap_stack, axis=0)
mean_df = pd.DataFrame(mean_matrix, columns=[f'ihg_{i}' for i in range(IHG_COUNT)])
mean_df.insert(0, 'pose_id', np.arange(POSE_COUNT))

# Plot mean heatmap
log_mean = mean_matrix.copy()
log_mean[log_mean == 0] = 1  # avoid log(0)

# for topic_heatmap in heatmap_stack:
#     diff = topic_heatmap - mean_matrix
#     diff_range_min = min(diff_range_min, diff.min())
#     diff_range_max = max(diff_range_max, diff.max())

plt.figure(figsize=(16, 8))
ax = sns.heatmap(
    log_mean,
    cmap=custom_cmap,
    norm=LogNorm(vmin=10, vmax=CLIP_MAX),
    cbar_kws={"label": "Mean Count (log scale)"}
)
ax.set_title("Mean Heatmap (All Topics)", fontsize=16)
ax.set_xlabel("ihg_id")
ax.set_ylabel("pose_id")

output_path = os.path.join(HEATMAP_FOLDER, "mean_heatmap.png")
plt.tight_layout()
plt.savefig(output_path)
plt.close()



# Loop through all CSV files
for topic_id in range(64):
    filename = f"topic_{topic_id}.csv"
    filepath = os.path.join(CSV_FOLDER, filename)
    
    if os.path.exists(filepath):
        print(f"Processing {filename}...")

        # Load CSV
        df = pd.read_csv(filepath)

        # Set up empty matrix for 32 poses × 128 gestures
        heatmap_matrix = np.zeros((POSE_COUNT, IHG_COUNT))

        # Ensure ihp_cluster is int (just in case)
        df['ihp_cluster'] = df['ihp_cluster'].astype(int)

        # # Filter out counts ≤ 100
        filtered_df = df.copy()
        gesture_cols = [f'ihg_{i}' for i in range(IHG_COUNT)]
        # filtered_df[gesture_cols] = filtered_df[gesture_cols].where(filtered_df[gesture_cols] > 100, 0)

        # Clip max values at CLIP_MAX
        clipped_df = filtered_df[gesture_cols].clip(upper=CLIP_MAX)

        # Now fill the heatmap matrix row by row
        for idx, row in df.iterrows():
            pose_id = int(row['ihp_cluster'])  # Make sure this is an integer
            if 0 <= pose_id < POSE_COUNT:
                heatmap_matrix[pose_id, :] = clipped_df.loc[idx].values

        # Plot heatmap
        plt.figure(figsize=(16, 8))
        # Use YlOrRd colormap with white for zero
        cmap = plt.get_cmap("YlOrRd")
        cmap_with_white = cmap(np.linspace(0, 1, 256))
        cmap_with_white[0] = [1, 1, 1, 1]  # Set 0 (lowest) to white
        custom_cmap = ListedColormap(cmap_with_white)

        # Avoid log(0) by setting zeros to a small value (e.g., 1)
        log_matrix = heatmap_matrix.copy()
        log_matrix[log_matrix == 0] = 1  # This will appear white due to colormap

        # Plot log heatmap
        ax = sns.heatmap(
            log_matrix,
            cmap=custom_cmap,
            norm=LogNorm(vmin=LOG_ZERO, vmax=CLIP_MAX),
            cbar_kws={"label": "Count (log scale)"}
        )
        ax.set_title(f"Heatmap for {filename}", fontsize=14)
        ax.set_xlabel("ihg_id")
        ax.set_ylabel("pose_id")

        # Save heatmap image
        output_filename = os.path.splitext(filename)[0] + "_heatmap.png"
        output_path = os.path.join(HEATMAP_FOLDER, output_filename)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


        # === DIFF FROM MEAN HEATMAP ===
        diff_matrix = heatmap_matrix - mean_matrix

        # Define your desired colors
        colors = [(1, 1, 0), (1, 1, 1), (0, 1, 0), (0, 0, 1), (1, 0, 1)]  # yellow → white → green → blue → magenta
        boundaries = [-300, 0, 1000, 2000, 4000]

        # Normalize to range [0, 1]
        min_val = min(boundaries)
        max_val = max(boundaries)
        norm_boundaries = [(b - min_val) / (max_val - min_val) for b in boundaries]

        # Make sure the boundaries start at 0 and end at 1
        # If they don't, we insert dummy control points
        if norm_boundaries[0] > 0:
            norm_boundaries = [0.0] + norm_boundaries
            colors = [colors[0]] + colors
        if norm_boundaries[-1] < 1:
            norm_boundaries = norm_boundaries + [1.0]
            colors = colors + [colors[-1]]

        # Create color dict
        cdict = {'red': [], 'green': [], 'blue': []}
        for i, color in enumerate(colors):
            pos = norm_boundaries[i]
            cdict['red'].append((pos, color[0], color[0]))
            cdict['green'].append((pos, color[1], color[1]))
            cdict['blue'].append((pos, color[2], color[2]))

        # Build custom colormap
        custom_cmap = LinearSegmentedColormap('custom_gradient', cdict)

        # Plot diff heatmap using the custom colormap
        plt.figure(figsize=(16, 8))
        ax = sns.heatmap(
            diff_matrix,
            cmap=custom_cmap,
            vmin=min(boundaries),
            vmax=max(boundaries),
            cbar_kws={"label": "Difference from Mean"}
        )
        ax.set_title(f"Difference from Mean Heatmap for {filename}", fontsize=14)
        ax.set_xlabel("ihg_id")
        ax.set_ylabel("pose_id")

        # Save diff heatmap
        diff_filename = os.path.splitext(filename)[0] + "_diff_from_mean_heatmap.png"
        diff_output_path = os.path.join(HEATMAP_FOLDER, "mean_difference", diff_filename)
        plt.tight_layout()
        plt.savefig(diff_output_path)
        plt.close()


# Collect significant diffs
diff_records = []

for topic_id in range(64):
    filename = f"topic_{topic_id}.csv"
    filepath = os.path.join(CSV_FOLDER, filename)

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['ihp_cluster'] = df['ihp_cluster'].astype(int)
        gesture_cols = [f'ihg_{i}' for i in range(IHG_COUNT)]
        clipped_df = df[gesture_cols].clip(upper=CLIP_MAX)

        # Build matrix
        heatmap_matrix = np.zeros((POSE_COUNT, IHG_COUNT))
        for idx, row in df.iterrows():
            pose_id = int(row['ihp_cluster'])
            if 0 <= pose_id < POSE_COUNT:
                heatmap_matrix[pose_id, :] = clipped_df.loc[idx].values

        # Compute diff from mean
        diff_matrix = heatmap_matrix - mean_matrix

        # Record all diffs > 500
        for pose_id in range(POSE_COUNT):
            for ihg_id in range(IHG_COUNT):
                diff_val = diff_matrix[pose_id, ihg_id]
                if diff_val > 500:
                    cell_count = heatmap_matrix[pose_id, ihg_id]
                    diff_records.append((topic_id, pose_id, ihg_id, diff_val, cell_count))

# Sort by diff descending
diff_records.sort(key=lambda x: x[3], reverse=True)

# Save to CSV
output_diff_csv = os.path.join(HEATMAP_FOLDER, "high_diff_cells.csv")
with open(output_diff_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["topic_id", "pose_id", "ihg_id", "diff", "cell_count"])
    writer.writerows(diff_records)