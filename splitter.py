import os
import shutil
import random

# Define paths for the original dataset and for train/test folders
folder_path = 'cardataset'
train_path = 'cardataset_train'
test_path = 'cardataset_test'

# Define the train-test split ratio
test_ratio = 0.2  # 20% of images for testing

# Create train and test directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Loop through each car model folder
for folder_name in os.listdir(folder_path):
    folder_full_path = os.path.join(folder_path, folder_name)
    
    if os.path.isdir(folder_full_path):
        # Create corresponding train/test subfolders for each model
        os.makedirs(os.path.join(train_path, folder_name), exist_ok=True)
        os.makedirs(os.path.join(test_path, folder_name), exist_ok=True)
        
        # List all images in the current folder and shuffle
        images = os.listdir(folder_full_path)
        random.shuffle(images)
        
        # Calculate split index
        split_idx = int(len(images) * (1 - test_ratio))
        
        # Split images into train and test sets
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Copy images to the train and test folders
        for image in train_images:
            shutil.copy(os.path.join(folder_full_path, image), os.path.join(train_path, folder_name, image))
        
        for image in test_images:
            shutil.copy(os.path.join(folder_full_path, image), os.path.join(test_path, folder_name, image))
        
        # Print out the count to verify split
        print(f"{folder_name}: {len(train_images)} images in train, {len(test_images)} images in test")

print("Images have been split into separate training and testing sets within each folder.")
