# test_vehicle_classifier.py

import torch
import pandas as pd
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import os

# Path to the saved model
model_path = 'saved_models/vehicle_classifier.pth'

# Load car labels
car_labels = pd.read_csv('names.csv').set_index('Label').to_dict()['Model']

# Define transformations (same as used during training)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the model
model = EfficientNet.from_name('efficientnet-b0')
num_classes = len(car_labels)
model._fc = torch.nn.Linear(model._fc.in_features, num_classes)  # Adjust output layer
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

print("Model loaded and set to evaluation mode.")

# Load a few test samples
test_data = pd.read_csv('anno_test.csv')
sample_data = test_data.sample(n=5)  # Select 5 random samples for testing

# Run inference on each sample
for index, row in sample_data.iterrows():
    img_path = row['Image']
    true_label = row['Label']
    
    # Load and preprocess the image
    try:
        image = Image.open(img_path).convert("RGB")
        image = test_transform(image).unsqueeze(0)  # Add batch dimension

        # Run the image through the model
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_label = predicted.item()

        # Get the predicted and true car model names
        predicted_model = car_labels[predicted_label]
        true_model = car_labels[true_label]
        print(f"Image: {img_path}")
        print(f"True Model: {true_model}, Predicted Model: {predicted_model}\n")

    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
