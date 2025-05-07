import cv2
import os
import matplotlib.pyplot as plt

# Root directory containing all the folders
root_dir = "directory"

# CLAHE parameters
clip_limit = 50
tile_grid_size = (30, 30)

# Loop through all subdirectories in root_dir
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        input_dir = folder_path
        output_dir = os.path.join(input_dir, f"CLAHE_{folder_name}")

        # Create the output folder if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Iterate through all TIFF files in the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith(".tif"):
                image_path = os.path.join(input_dir, filename)
                image = cv2.imread(image_path)

                # Convert to grayscale
                grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                final_img = clahe.apply(grayimg)

                # Save the enhanced image
                output_path = os.path.join(output_dir, filename.replace(".tif", "_CLAHE.tif"))
                cv2.imwrite(output_path, final_img)

        print(f"Enhancement complete for folder: {folder_name}. Output saved to: {output_dir}")
