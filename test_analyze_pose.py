import pandas as pd
import os
import shutil
from unittest.mock import MagicMock

# Create dummy SortPose class to isolate analyze_head_pose_by_cluster
class SortPoseMock:
    def __init__(self):
        self.output_dir = 'analysis/face_angle'
        self.counter_dict = {
            '1': {'1': 10, '2': 5},
            '2': {'1': 20}
        }
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_head_pose_by_cluster(self, df):
        # Implementation of the core logic to be tested
        stats_list = []
        for cluster_no, poses in self.counter_dict.items():
            for pose_no, count in poses.items():
                mask = (df['cluster_no'] == cluster_no) & (df['pose_no'] == pose_no)
                subset = df[mask]
                
                if not subset.empty:
                    yaw_vals = subset['yaw'].abs()
                    survival_pct = (yaw_vals <= 15).mean() * 100
                    
                    stats_list.append({
                        'cluster_no': cluster_no,
                        'pose_no': pose_no,
                        'n': count,
                        'yaw_survival_pct': survival_pct,
                        'yaw_std': subset['yaw'].std(),
                        'pitch_std': subset['pitch'].std()
                    })
        
        stats_df = pd.DataFrame(stats_list)
        output_file = os.path.join(self.output_dir, 'head_pose_filter_comparison.csv')
        
        if os.path.exists(output_file):
            stats_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            stats_df.to_csv(output_file, index=False)
        return output_file

# Setup tiny synthetic data
data = {
    'cluster_no': ['1', '1', '1', '2', '2'],
    'pose_no': ['1', '1', '2', '1', '1'],
    'pitch': [1, 2, 3, 4, 5],
    'yaw': [0, 20, 5, 30, 40],
    'roll': [0, 0, 0, 0, 0]
}
df = pd.DataFrame(data)

# Run check
comparitor_file = 'analysis/face_angle/head_pose_filter_comparison.csv'
if os.path.exists(comparitor_file):
    os.remove(comparitor_file)

sorter = SortPoseMock()
out_path = sorter.analyze_head_pose_by_cluster(df)

if os.path.exists(out_path):
    result_df = pd.read_csv(out_path)
    print(f"File written: {out_path}")
    print(f"Rows written: {len(result_df)}")
    print(result_df.to_string())
else:
    print("Failed to write file.")
