import os

# Define the paths for training and testing folders
train_path = 'cardataset_train'
test_path = 'cardataset_test'

# Count the number of folders in each directory
train_folder_count = len([name for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name))])
test_folder_count = len([name for name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, name))])

print(f"Number of folders in '{train_path}': {train_folder_count}")
print(f"Number of folders in '{test_path}': {test_folder_count}")