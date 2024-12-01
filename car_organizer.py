import os
import shutil
import re

# Define the folder path
folder_path = 'cardataset'

# Updated regular expression pattern to match Make_Model_Year with various separators
pattern = r'^([A-Za-z0-9]+(?:[_ -][A-Za-z0-9]+)*_\d{4})'

# Check if the folder exists
if not os.path.isdir(folder_path):
    print(f"Folder '{folder_path}' does not exist.")
else:
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file, not a directory
        if os.path.isfile(file_path):
            print(f"Processing file: {filename}")
            
            # Attempt to match the Make_Model_Year pattern at the beginning of the filename
            match = re.match(pattern, filename)
            if match:
                # Extract the Make_Model_Year part, replacing spaces/hyphens with underscores
                make_model_year = match.group(1).replace(' ', '_').replace('-', '_')
                print(f"Matched pattern: {make_model_year}")
                
                # Define the directory for this Make_Model_Year
                target_folder = os.path.join(folder_path, make_model_year)
                
                # Create the directory if it doesn't exist
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                    print(f"Created directory: {target_folder}")
                
                # Move the file into the target folder
                try:
                    shutil.move(file_path, os.path.join(target_folder, filename))
                    print(f"Moved {filename} to {make_model_year}")
                except Exception as e:
                    print(f"Failed to move {filename} to {make_model_year}: {e}")
            else:
                print(f"No match for pattern in filename: {filename}")
        else:
            print(f"Skipping directory: {filename}")

print("Organizing process complete.")
