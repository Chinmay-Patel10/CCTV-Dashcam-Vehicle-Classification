import os

# Define the folder path and the list of keywords
folder_path = 'cardataset'
keywords = [
   "Rolls", "Ferrari", "Porsche", "Aston", "Bentley",
   "Alfa", "FIAT", "Jaguar", "Lamborghini",
   "Rover", "Maserati", "smart"
]

# Track the deletion process with debug statements
for filename in os.listdir(folder_path):
    try:
        # Check if any keyword is in the filename (case-sensitive)
        if any(keyword in filename for keyword in keywords):
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)  # Delete the file
            print(f"Deleted: {filename}")
        else:
            print(f"No match: {filename}")
    except Exception as e:
        print(f"Failed to delete {filename}: {e}")

print("Deletion process complete.")



