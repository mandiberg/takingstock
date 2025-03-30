from PIL import Image
import os
import csv
import sys
import math

# Add facemap to the path
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from mp_db_io import DataIO
IS_SSD = True

def trim_bottom(img, file_path):
    trim = 0
    if "images_shutterstock" in file_path: trim = 100
    elif "images_alamy" in file_path: trim = 90
    if trim > 0:
        print(f"Trimming {trim} pixels from the bottom of {file_path}")
        img = img.crop((0, 0, img.width, img.height - trim))
    return img

def create_image_grids(file_paths, row_height=700, max_row_width=8000, rows_per_grid=7):
    """
    Creates composite image grids from a list of file paths.
    
    Args:
        file_paths (list): List of paths to image files
        row_height (int): Height of each row in pixels
        max_row_width (int): Maximum width of each row in pixels
        rows_per_grid (int): Number of rows per composite grid
    
    Returns:
        int: Number of composite grids created
    """
    if not file_paths:
        return 0
    
    # Initialize variables
    current_images = []  # Images for the current grid
    row_images = []      # Images for the current row
    current_row_width = 0
    current_row = 0
    grid_count = 0
    
    # Process each file path
    for file_path in file_paths:
        try:
            # Open image and calculate its dimensions when resized to row height
            img = Image.open(file_path)
            img = trim_bottom(img, file_path)
            aspect_ratio = img.width / img.height
            new_width = int(row_height * aspect_ratio)
            
            # Check if adding this image would exceed the maximum row width
            if current_row_width + new_width > max_row_width and row_images:
                # Create a row image by concatenating horizontally
                row_img = create_row(row_images, row_height)
                current_images.append(row_img)
                
                # Reset row variables
                row_images = []
                current_row_width = 0
                current_row += 1
                
                # Check if we've filled all rows for the current grid
                if current_row >= rows_per_grid:
                    # Create and save the completed grid
                    create_grid(current_images, grid_count)
                    grid_count += 1
                    
                    # Reset grid variables
                    current_images = []
                    current_row = 0
            
            # Resize the image and add it to the current row
            resized_img = img.resize((new_width, row_height), Image.Resampling.LANCZOS)
            row_images.append(resized_img)
            current_row_width += new_width
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Handle any remaining images
    if row_images:
        row_img = create_row(row_images, row_height)
        current_images.append(row_img)
    
    # Create the final grid if there are any images left
    if current_images:
        create_grid(current_images, grid_count)
        grid_count += 1
    
    return grid_count

def create_row(images, row_height):
    """
    Creates a single row by concatenating images horizontally.
    
    Args:
        images (list): List of PIL Image objects
        row_height (int): Height of the row in pixels
    
    Returns:
        PIL.Image: Composite row image
    """
    # Calculate the total width of the row
    total_width = sum(img.width for img in images)
    
    # Create a new blank image for the row
    row_img = Image.new('RGB', (total_width, row_height))
    
    # Paste each image into the row
    x_offset = 0
    for img in images:
        row_img.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return row_img

def create_grid(row_images, grid_number):
    """
    Creates a grid from row images and saves it to a file.
    
    Args:
        row_images (list): List of row images (PIL Image objects)
        grid_number (int): Index of the grid for filename
    """
    # Calculate the dimensions of the grid
    max_width = max(img.width for img in row_images)
    total_height = sum(img.height for img in row_images)
    
    # Create a new blank image for the grid
    grid_img = Image.new('RGB', (max_width, total_height))
    
    # Paste each row into the grid
    y_offset = 0
    for row_img in row_images:
        grid_img.paste(row_img, (0, y_offset))
        y_offset += row_img.height
    
    # Save the grid to a file
    output_dir = "output_grids"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"composite_grid_{grid_number:03d}.jpg")
    grid_img.save(output_path, quality=95)
    print(f"Created composite grid: {output_path}")

def read_csv_and_build_paths():
    """
    Reads the CSV file and builds full file paths using the DataIO class.
    
    Returns:
        list: Full paths to image files
    """
    # Initialize DataIO to get access to folder paths
    io = DataIO(IS_SSD)
    
    # Path to the CSV file
    csv_path = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/image_grids/HEFTbook/image_file_list.csv'
    
    # List to store full file paths
    full_paths = []
    
    # Read the CSV file
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        #skip the first row
        next(reader)
        
        for row in reader:
            print(f"Processing row: {row}")
            try:
                # Get site_name_id and imagename from the row
                site_id = int(row['site_name_id'])
                image_name = row['imagename']
                print(f"Processing site_id: {site_id}, image_name: {image_name}")
                # Get the base folder path from DataIO
                base_folder = io.folder_list[site_id]
                
                # Construct the full path to the image
                full_path = os.path.join(base_folder, image_name)
                
                # Check if the file exists
                if os.path.isfile(full_path):
                    full_paths.append(full_path)
                else:
                    print(f"Warning: File not found: {full_path}")
            except Exception as e:
                print(f"Error processing row {row}: {e}")
    
    print(f"Found {len(full_paths)} valid image paths")
    return full_paths

def main():
    # Get list of image file paths from the CSV
    file_paths = read_csv_and_build_paths()
    
    # Create the image grids
    num_grids = create_image_grids(
        file_paths, 
        row_height=700, 
        max_row_width=8000, 
        rows_per_grid=7
    )
    
    print(f"Completed processing. Created {num_grids} composite grids.")

if __name__ == "__main__":
    main()