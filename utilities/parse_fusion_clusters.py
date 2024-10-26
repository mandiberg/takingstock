import pandas as pd
import glob
import os

# Define the root folder path where the topic CSV files are stored
ROOT_FOLDER_PATH = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/october_fusion_clusters'

# Define the output CSV file path
output_csv = os.path.join(ROOT_FOLDER_PATH, 'count_of_fusion_clusters.csv')

# Initialize an empty list to store results
results = []

# Read all topic CSV files from the root folder
for topic_id in range(64):
    csv_file_path = os.path.join(ROOT_FOLDER_PATH, f"topic_{topic_id}.csv")
    
    # Check if the file exists
    if os.path.exists(csv_file_path):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        
        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            pose_id = row['ihp_cluster']
            
            # Check for counts greater than 100 in each ihg column
            for ihg_id in range(128):  # Assuming there are ihg_0 to ihg_127 columns
                count = row[f'ihg_{ihg_id}']
                
                if count >= 100:
                    # Append the filtered results to the list
                    results.append({
                        'topic_id': topic_id,
                        'pose_id': pose_id,
                        'ihg_id': ihg_id,
                        'count': count
                    })

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to count.csv
results_df.to_csv(output_csv, index=False)

print(f"Output saved to {output_csv}")
