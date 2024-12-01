import os
import pandas as pd
import random

# Define paths
folder_path = 'cardataset'
train_path = 'cardataset_train'
test_path = 'cardataset_test'

# Generate 'names.csv'
car_models = sorted(os.listdir(folder_path))
car_labels = {model: idx for idx, model in enumerate(car_models)}
names_df = pd.DataFrame(list(car_labels.items()), columns=['Model', 'Label'])
names_df.to_csv('names.csv', index=False)
print("names.csv created")

# Function to generate annotation CSVs
def generate_annotations(image_dir, csv_filename):
    data = []
    for model in os.listdir(image_dir):
        model_path = os.path.join(image_dir, model)
        if os.path.isdir(model_path):
            label = car_labels[model]  # Get label for each model
            for image in os.listdir(model_path):
                image_path = os.path.join(model_path, image)
                data.append([image_path, label])
    df = pd.DataFrame(data, columns=['Image', 'Label'])
    df.to_csv(csv_filename, index=False)
    print(f"{csv_filename} created")

# Generate 'anno_train.csv' and 'anno_test.csv'
generate_annotations(train_path, 'anno_train.csv')
generate_annotations(test_path, 'anno_test.csv')
