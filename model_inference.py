import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from efficientnet_pytorch import EfficientNet

# Load car labels
car_labels = pd.read_csv('names.csv').set_index('Label').to_dict()['Model']

# Define transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the classification model
model = EfficientNet.from_name('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, len(car_labels))
model.load_state_dict(torch.load('saved_models/vehicle_classifier.pth', map_location='cpu'))
model.eval()  # Set model to evaluation mode

def predict_top_k(image_path, k=5):
    image = Image.open(image_path).convert("RGB")
    image = test_transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(image)
        _, topk_indices = torch.topk(output, k)
    
    topk_predictions = []
    for idx in topk_indices[0]:
        label = idx.item()
        car_model = car_labels.get(label, "Unknown")
        topk_predictions.append(car_model)
    
    return topk_predictions
