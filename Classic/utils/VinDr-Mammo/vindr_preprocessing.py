import os
from PIL import Image, ImageDraw
import pandas as pd

# Define paths
base_images_dir = 'C:\\Users\\Win10\\Desktop\\scientificResearch\\cancerResearch\\data\\png_files'
train_cropped_output_dir = 'C:\\Users\\Win10\\Desktop\\scientificResearch\\cancerResearch\\data\\png_files\\cropped\\train'
test_cropped_output_dir = 'C:\\Users\\Win10\\Desktop\\scientificResearch\\cancerResearch\\data\\png_files\\cropped\\test'

# Ensure output base directories exist
os.makedirs(train_cropped_output_dir, exist_ok=True)
os.makedirs(test_cropped_output_dir, exist_ok=True)

# Load metadata and annotations
metadata_path = 'C:\\Users\\Win10\\Desktop\\scientificResearch\\cancerResearch\\data\\csv\\metadata.csv'
finding_annotations_path = 'C:\\Users\\Win10\\Desktop\\scientificResearch\\cancerResearch\\data\\csv\\finding_annotations.csv'
breast_level_annotations_path = 'C:\\Users\\Win10\\Desktop\\scientificResearch\\cancerResearch\\data\\csv\\breast-level_annotations.csv'

metadata = pd.read_csv(metadata_path)
finding_annotations = pd.read_csv(finding_annotations_path)
breast_level_annotations = pd.read_csv(breast_level_annotations_path)

# Define labels ( 0 for BI-RADS 1/BI-RADS 2 and 1 for BI-RADS 3/BI-RADS 4/BI-RADS 5 )
breast_level_annotations["breast_birads"].replace(["BI-RADS 1","BI-RADS 2"], 0, inplace=True)
breast_level_annotations["breast_birads"].replace(["BI-RADS 3","BI-RADS 4", "BI-RADS 5"], 1, inplace=True)

# Define the size of the square to resize to
square_size = 224

# Process each image in train and test directories
for folder in ['train','test']:
    images_dir = os.path.join(base_images_dir, folder)

    for _, row in breast_level_annotations.iterrows():
        image_name = row['image_id']
        laterality = row['laterality']
        split = row['split']
        label = row['breast_birads']

        # Determine the output directory based on split
        output_dir = os.path.join(train_cropped_output_dir if split == 'training' else test_cropped_output_dir, str(label))
        os.makedirs(output_dir, exist_ok=True)

        image_path = os.path.join(images_dir, image_name + '.png')
        if not os.path.exists(image_path):
            continue

        # Load image
        image = Image.open(image_path)
        width, height = image.size

        # Determine the size of the width crop (three quarters of the width)
        crop_width = width * 3 // 4
        # Determine the size of the height crop (three quarters of the height)
        crop_height = height * 3 // 4

        # Crop image based on laterality for width
        if laterality == 'R':
            left = width // 4
            right = width
        elif laterality == 'L':
            left = 0
            right = crop_width
        else:
            print(f"Unknown laterality for image {image_name}. Skipping.")
            continue

        # Crop the image
        cropped_image = image.crop((left, 0, right, crop_height))

        # Resize the cropped image to a square
        square_image = cropped_image.resize((square_size, square_size))

        # Save the cropped and resized image
        output_image_path = os.path.join(output_dir, image_name + '.png')
        square_image.save(output_image_path)

        print(f"Cropped and resized image saved: {output_image_path}")

print("Cropping, resizing, and annotation adjustment completed.")
