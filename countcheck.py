import os

# Define the folder path
folder_path = 'cardataset'

# Count the number of files in the folder
file_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])

print(f"Number of files in '{folder_path}': {file_count}")


