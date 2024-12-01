# train_vehicle_classifier.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from car_dataset import CarDataset, train_transform, test_transform
import pandas as pd
import time
import os

# Load car labels
try:
    car_labels = pd.read_csv('names.csv').set_index('Model').to_dict()['Label']
    print("Loaded class labels from names.csv")
except Exception as e:
    print(f"Error loading names.csv: {e}")

# Initialize datasets and dataloaders
try:
    train_dataset = CarDataset('anno_train.csv', transform=train_transform)
    test_dataset = CarDataset('anno_test.csv', transform=test_transform)
    print("Initialized training and testing datasets.")
except Exception as e:
    print(f"Error initializing datasets: {e}")

# Define a custom collate function to skip None values
def collate_fn(batch):
    # Filter out any None values in the batch
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if len(batch) == 0:
        return None, None
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
print("Data loaders created.")

# Load EfficientNet model and modify the output layer
try:
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, len(car_labels))
    print("EfficientNet model loaded and output layer adjusted.")
except Exception as e:
    print(f"Error loading or configuring the model: {e}")

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Loss function and optimizer initialized.")

# Directory for saving model checkpoints
model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)

# Helper function to save model checkpoints
def save_checkpoint(epoch, model, optimizer, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

# Training loop with error handling for corrupted images, time tracking, and checkpointing
num_epochs = 10
checkpoint_interval = 5  # Save checkpoint every 5 epochs

try:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()  # Start time for the epoch
        print(f"Starting epoch {epoch+1}/{num_epochs}...")

        for i, (images, labels) in enumerate(train_loader):
            # Skip invalid batches (empty batches)
            if images is None or labels is None or len(images) == 0:
                print(f"Skipped invalid batch at step {i+1} in epoch {epoch+1}")
                continue

            optimizer.zero_grad()

            # Forward pass
            try:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if (i + 1) % 10 == 0:  # Print loss every 10 batches
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Error during training at epoch {epoch+1}, batch {i+1}: {e}")
                # Save a checkpoint in case of an error
                checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}_step_{i+1}.pth')
                save_checkpoint(epoch+1, model, optimizer, running_loss / (i+1), checkpoint_path)
                raise  # Re-raise the exception to stop training

        # Calculate and print epoch duration and estimated remaining time
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_duration:.2f} seconds with average loss: {running_loss/len(train_loader):.4f}")

        # Save checkpoint every few epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(epoch+1, model, optimizer, running_loss / len(train_loader), checkpoint_path)

        # Estimated time left for remaining epochs
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_time_left = epoch_duration * remaining_epochs
        print(f"Estimated time left for training: {estimated_time_left / 60:.2f} minutes")

    # Final model save
    final_model_path = os.path.join(model_dir, 'vehicle_classifier.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved as {final_model_path}")

except Exception as e:
    print(f"Training interrupted due to an error: {e}")
    # Save a final checkpoint in case of an unexpected crash
    crash_checkpoint_path = os.path.join(model_dir, 'crash_checkpoint.pth')
    save_checkpoint(epoch+1, model, optimizer, running_loss / (i+1), crash_checkpoint_path)
