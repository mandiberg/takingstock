import pandas as pd
import matplotlib.pyplot as plt
import os
import ast # For safely evaluating string representations of lists/tuples

# Define the folder path for CSV files.
# IMPORTANT: Please update this variable with the actual path to your CSV files.
CSV_FOLDER_PATH = "/Users/tenchc/Documents/GitHub/taking_stock_production/bbox_calc" # Example: "/Users/youruser/Documents/my_csv_data"
OUTPUT_DIR = "/Users/tenchc/Documents/GitHub/taking_stock_production/bbox_output" # Centralized output directory for graphs

def parse_bbox_string(bbox_str):
    """Parses a bbox string like '[x, y, w, h]' into a tuple of floats."""
    if pd.isna(bbox_str):
        return None
    try:
        # Use ast.literal_eval for safe evaluation of string literals
        bbox_list = ast.literal_eval(bbox_str)
        if isinstance(bbox_list, (list, tuple)) and len(bbox_list) == 4:
            return tuple(map(float, bbox_list))
    except (ValueError, SyntaxError):
        return None
    return None

def calculate_iou(boxA, boxB):
    """
    Calculates Intersection over Union (IoU) for two bounding boxes.
    Boxes are expected in (x, y, w, h) format.
    Returns 0 if any box is None or has zero or negative area.
    """
    if boxA is None or boxB is None:
        return 0.0
    if boxA[2] <= 0 or boxA[3] <= 0 or boxB[2] <= 0 or boxB[3] <= 0: # Check for zero or negative width/height
        return 0.0

    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    boxA_x1, boxA_y1, boxA_w, boxA_h = boxA
    boxA_x2, boxA_y2 = boxA_x1 + boxA_w, boxA_y1 + boxA_h

    boxB_x1, boxB_y1, boxB_w, boxB_h = boxB
    boxB_x2, boxB_y2 = boxB_x1 + boxB_w, boxB_y1 + boxB_h

    # Determine the coordinates of the intersection rectangle
    xA = max(boxA_x1, boxB_x1)
    yA = max(boxA_y1, boxB_y1)
    xB = min(boxA_x2, boxB_x2)
    yB = min(boxA_y2, boxB_y2)

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = boxA_w * boxA_h
    boxBArea = boxB_w * boxB_h

    # Compute the union area
    unionArea = float(boxAArea + boxBArea - interArea)

    # Handle case where unionArea is zero (e.g., both boxes are points or lines)
    if unionArea == 0:
        return 0.0

    iou = interArea / unionArea
    return iou

def generate_graphs_for_csv(filepath):
    """
    Generates informational graphs comparing original and new bounding boxes
    from a single CSV file, grouped by cluster if a 'cluster_no' column exists.
    All graphs for a cluster are saved into a single PNG.

    Assumes the CSV contains columns 'bbox' (original) and 'new_bbox' (new),
    where bounding boxes are stored as string representations of lists, e.g., "[x, y, w, h]".
    An optional 'cluster_no' column can be used for grouping.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {filepath}")
        return
    except Exception as e:
        print(f"Error reading CSV file {filepath}: {e}")
        return

    filename = os.path.basename(filepath)
    base_filename = os.path.splitext(filename)[0]

    print(f"Processing {filename}...")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define expected bbox columns based on the new CSV format (only top/bottom)
    required_orig_bbox_cols = ['original bbox.top', 'original bbox.bottom']
    required_new_bbox_cols = ['new bbox.top', 'new bbox.bottom']
    required_norm_diff_cols = ['normalized diff top', 'normalized diff bottom']
    all_required_bbox_cols = required_orig_bbox_cols + required_new_bbox_cols + required_norm_diff_cols

    if not all(col in df.columns for col in all_required_bbox_cols):
        print(f"Skipping {filename}: Missing one or more required bbox columns ({all_required_bbox_cols}).")
        return

    # Drop rows where any of the required bbox columns have NaN values
    df.dropna(subset=all_required_bbox_cols, inplace=True)

    if df.empty:
        print(f"No valid bbox data found in {filename} after checking required columns. Skipping.")
        return

    # Extract individual coordinates for original bounding boxes (only top/bottom)
    df['orig_top'] = df['original bbox.top']
    df['orig_bottom'] = df['original bbox.bottom']
    df['orig_height'] = df['original bbox.bottom'] - df['original bbox.top']

    # Extract individual coordinates for new bounding boxes (only top/bottom)
    df['new_top'] = df['new bbox.top']
    df['new_bottom'] = df['new bbox.bottom']
    df['new_height'] = df['new bbox.bottom'] - df['new bbox.top']

    # Filter out rows with invalid bounding box dimensions (height <= 0)
    df = df[(df['orig_height'] > 0) & (df['new_height'] > 0)]

    if df.empty:
        print(f"No valid bbox data found in {filename} after deriving dimensions. Skipping.")
        return

    # Calculate comparison metrics (only top/bottom)
    df['diff_top'] = df['new_top'] - df['orig_top']
    df['diff_bottom'] = df['new_bottom'] - df['orig_bottom']
    df['diff_height'] = df['new_height'] - df['orig_height']
    df['orig_height'] = df['orig_height']
    df['new_height'] = df['new_height']

    # Group by 'cluster_no' if the column exists, otherwise treat the whole file as one group
    if 'cluster_no' in df.columns:
        groups = df.groupby('cluster_no')
        print(f"Found {len(groups)} clusters in {filename}.")
        # Convert groupby object to dictionary for iteration
        groups_dict = {name: group for name, group in groups}
    else:
        groups_dict = {'all_data': df} # Treat entire file as one group
        print(f"No 'cluster_no' column found in {filename}. Generating graphs for the entire file.")

    for cluster_name, group_df in groups_dict.items():
        if group_df.empty:
            print(f"Cluster {cluster_name} is empty, skipping.")
            continue

        print(f"Generating graphs for Cluster: {cluster_name} (Count: {len(group_df)})")

        # Create a single figure with multiple subplots (2 rows, 2 columns)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten() # Flatten for easy iteration

        # --- Plot 1: Difference in Top and Bottom positions (Scatter) ---
        axes[0].scatter(group_df['diff_top'], group_df['diff_bottom'], alpha=0.6, color='purple')
        axes[0].set_title(f'Bbox Position Shift (Cluster: {cluster_name})')
        axes[0].set_xlabel('Difference in Top (New - Original)')
        axes[0].set_ylabel('Difference in Bottom (New - Original)')
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # --- Plot 2: Height Comparison (Box Plot) ---
        data_to_plot_height = [
            group_df['orig_height'].dropna(), group_df['new_height'].dropna()
        ]
        labels_height = ['Original Height', 'New Height']
        axes[1].boxplot(data_to_plot_height, labels=labels_height, patch_artist=True, medianprops={'color': 'black'})
        axes[1].set_title(f'Height Comparison (Cluster: {cluster_name})')
        axes[1].set_ylabel('Height Value (Pixels)')
        axes[1].grid(axis='y', alpha=0.75)

        # --- Plot 3: Histogram of Height Difference ---
        axes[2].hist(group_df['diff_height'].dropna(), bins=20, color='lightgreen', edgecolor='black')
        axes[2].set_title(f'Distribution of Height Difference (Cluster: {cluster_name})')
        axes[2].set_xlabel('Height Difference (New - Original)')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(axis='y', alpha=0.75)

        # --- Plot 4: Top vs Bottom Position Changes ---
        axes[3].scatter(group_df['orig_top'], group_df['new_top'], alpha=0.6, color='red', label='Top')
        axes[3].scatter(group_df['orig_bottom'], group_df['new_bottom'], alpha=0.6, color='blue', label='Bottom')
        axes[3].set_title(f'Position Comparison (Cluster: {cluster_name})')
        axes[3].set_xlabel('Original Position')
        axes[3].set_ylabel('New Position')
        axes[3].legend()
        axes[3].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        # Sanitize cluster_name for filename if it's not a simple number
        safe_cluster_name = str(cluster_name).replace(os.sep, '_').replace(':', '_')
        output_path = os.path.join(OUTPUT_DIR, f"{base_filename}_cluster_{safe_cluster_name}_bbox_comparison.png")
        plt.savefig(output_path)
        plt.close(fig) # Close the figure to free memory
        print(f"Saved comparison graphs for cluster {cluster_name} to {output_path}")

    # --- Print descriptive statistics for the entire file ---
    print(f"\nDescriptive Statistics for {filename} (All Data):")
    print(df[['diff_top', 'diff_bottom', 'diff_height', 'orig_height', 'new_height']].describe())
    print("-" * 50)


def generate_cluster_comparison_graphs(csv_folder_path):
    """
    Generates graphs comparing normalized differences between clusters across all CSV files.
    Each CSV file represents one cluster, so we aggregate data from all CSV files.
    """
    if not os.path.isdir(csv_folder_path):
        print(f"Error: The specified folder path does not exist: {csv_folder_path}")
        return

    csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {csv_folder_path}")
        return

    print(f"Processing cluster comparison across {len(csv_files)} CSV files...")

    # Collect data from all CSV files
    all_data = []
    cluster_info = []
    
    for csv_file in csv_files:
        filepath = os.path.join(csv_folder_path, csv_file)
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading CSV file {csv_file}: {e}")
            continue

        # Check for required columns
        required_cols = ['cluster_no', 'normalized diff top', 'normalized diff bottom']
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {csv_file}: Missing required columns ({required_cols}).")
            continue

        # Drop rows with NaN values in the required columns
        df_clean = df.dropna(subset=required_cols)
        
        if df_clean.empty:
            print(f"No valid data found in {csv_file}. Skipping.")
            continue

        # Add filename info to identify the cluster
        df_clean['filename'] = csv_file
        all_data.append(df_clean)
        
        # Store cluster info
        cluster_no = df_clean['cluster_no'].iloc[0]  # All rows should have same cluster_no
        cluster_info.append({
            'cluster_no': cluster_no,
            'filename': csv_file,
            'count': len(df_clean)
        })

    if not all_data:
        print("No valid data found across all CSV files.")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Calculate median normalized differences per cluster
    cluster_stats = combined_df.groupby('cluster_no').agg({
        'normalized diff top': 'median',
        'normalized diff bottom': 'median',
        'image_id': 'count'  # Count of images per cluster
    }).rename(columns={'image_id': 'count'})

    if len(cluster_stats) < 2:
        print(f"Need at least 2 clusters for comparison. Found {len(cluster_stats)} clusters across all files.")
        return

    # Create cluster comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # --- Plot 1: Median Normalized Differences by Cluster (Bar Plot) ---
    clusters = cluster_stats.index
    x_pos = range(len(clusters))
    
    axes[0, 0].bar([x - 0.2 for x in x_pos], cluster_stats['normalized diff top'], 
                   width=0.4, label='Top', alpha=0.7, color='skyblue')
    axes[0, 0].bar([x + 0.2 for x in x_pos], cluster_stats['normalized diff bottom'], 
                   width=0.4, label='Bottom', alpha=0.7, color='orange')
    axes[0, 0].set_xlabel('Cluster Number')
    axes[0, 0].set_ylabel('Median Normalized Difference')
    axes[0, 0].set_title('Median Normalized Differences by Cluster (All CSV Files)')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(clusters)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    # --- Plot 2: Scatter Plot of Top vs Bottom Normalized Differences ---
    colors = plt.cm.tab10(range(len(clusters)))
    for i, cluster in enumerate(clusters):
        cluster_data = combined_df[combined_df['cluster_no'] == cluster]
        axes[0, 1].scatter(cluster_data['normalized diff top'], 
                          cluster_data['normalized diff bottom'], 
                          alpha=0.6, color=colors[i])
    
    axes[0, 1].set_xlabel('Normalized Difference Top')
    axes[0, 1].set_ylabel('Normalized Difference Bottom')
    axes[0, 1].set_title('Normalized Differences: Top vs Bottom by Cluster')
    axes[0, 1].grid(True, alpha=0.3)

    # --- Plot 3: Box Plot of Normalized Differences by Cluster ---
    top_data = [combined_df[combined_df['cluster_no'] == cluster]['normalized diff top'].values 
                for cluster in clusters]
    bottom_data = [combined_df[combined_df['cluster_no'] == cluster]['normalized diff bottom'].values 
                   for cluster in clusters]
    
    bp1 = axes[1, 0].boxplot(top_data, labels=[f'C{c}' for c in clusters], 
                             patch_artist=True, positions=range(1, len(clusters)+1))
    bp2 = axes[1, 0].boxplot(bottom_data, labels=[f'C{c}' for c in clusters], 
                             patch_artist=True, positions=[x + 0.3 for x in range(1, len(clusters)+1)])
    
    # Color the boxes
    for patch in bp1['boxes']:
        patch.set_facecolor('skyblue')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('orange')
        patch.set_alpha(0.7)
    
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Normalized Difference')
    axes[1, 0].set_title('Distribution of Normalized Differences by Cluster')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Add legend for box plots
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='skyblue', alpha=0.7, label='Top'),
                      Patch(facecolor='orange', alpha=0.7, label='Bottom')]
    axes[1, 0].legend(handles=legend_elements)

    # --- Plot 4: Top vs Bottom Normalized Difference Comparison ---
    axes[1, 1].scatter(cluster_stats['normalized diff top'], cluster_stats['normalized diff bottom'], 
                      alpha=0.7, s=100, color='purple')
    
    # Add cluster labels to points
    for cluster, row in cluster_stats.iterrows():
        axes[1, 1].annotate(f'C{cluster}', (row['normalized diff top'], row['normalized diff bottom']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add diagonal line for reference (perfect correlation)
    max_val = max(cluster_stats['normalized diff top'].max(), cluster_stats['normalized diff bottom'].max())
    min_val = min(cluster_stats['normalized diff top'].min(), cluster_stats['normalized diff bottom'].min())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Correlation')
    
    axes[1, 1].set_xlabel('Median Normalized Difference Top')
    axes[1, 1].set_ylabel('Median Normalized Difference Bottom')
    axes[1, 1].set_title('Top vs Bottom Accuracy by Cluster')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save the cluster comparison graph
    output_path = os.path.join(OUTPUT_DIR, "all_clusters_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved cluster comparison graphs to {output_path}")
    
    # Print cluster statistics summary
    print(f"\nCluster Statistics Summary (All CSV Files):")
    print(cluster_stats.round(4))
    print("-" * 50)


def main():
    """
    Main function to iterate through CSV files in the specified folder
    and generate graphs for each.
    """
    if not CSV_FOLDER_PATH:
        print("Error: Please set the 'CSV_FOLDER_PATH' variable to the directory containing your CSV files.")
        return

    if not os.path.isdir(CSV_FOLDER_PATH):
        print(f"Error: The specified folder path does not exist: {CSV_FOLDER_PATH}")
        return

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_files = [f for f in os.listdir(CSV_FOLDER_PATH) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in {CSV_FOLDER_PATH}")
        return

    for csv_file in csv_files:
        filepath = os.path.join(CSV_FOLDER_PATH, csv_file)
        generate_graphs_for_csv(filepath)
    
    # Generate cluster comparison graphs using all CSV files
    generate_cluster_comparison_graphs(CSV_FOLDER_PATH)

if __name__ == "__main__":
    main()
