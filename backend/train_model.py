import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import time

print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_FILENAME = 'mnist_pytorch_cnn_v2.pth'
DATA_DIR = './data'
USE_TORCH_COMPILE = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
print("Loading MNIST dataset...")
os.makedirs(DATA_DIR, exist_ok=True)
train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True) # Added workers/pin_memory
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)
print("Dataset loaded.")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(320, 50) # 20 * 4 * 4 = 320
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to(device)
print("\nModel Architecture:")
print(model)

if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
    print("\nAttempting to compile model with torch.compile()...")
    compile_mode = "default"
    try:
        start_time = time.time()
        model = torch.compile(model, mode=compile_mode)
        end_time = time.time()
        print(f"Model compiled successfully with mode='{compile_mode}' in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Model compilation failed (continuing without compilation): {e}")
        USE_TORCH_COMPILE = False
else:
    if USE_TORCH_COMPILE:
        print("\ntorch.compile() not available or disabled. Proceeding without compilation.")
    USE_TORCH_COMPILE = False


# --- Loss and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training ---
print("\n--- Starting Training ---")
model.train()
train_start_time = time.time()
for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()

        if (batch_idx + 1) % 100 == 0:
             print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_end_time = time.time()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct_train / total_train
    print(f'\nEpoch {epoch+1} Summary: Avg Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Time: {epoch_end_time - epoch_start_time:.2f}s')

train_end_time = time.time()
print(f"--- Finished Training (Total Time: {train_end_time - train_start_time:.2f}s) ---")

# --- Evaluation Loop ---
print("\n--- Evaluating Model on Test Set ---")
model.eval()
test_loss = 0
correct_test = 0
total_test = 0
eval_start_time = time.time()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        outputs = model(data)
        test_loss += criterion(outputs, target).item()
        _, predicted = torch.max(outputs.data, 1)
        total_test += target.size(0)
        correct_test += (predicted == target).sum().item()

eval_end_time = time.time()
test_loss /= len(test_loader)
test_acc = 100. * correct_test / total_test
print(f'\nTest Set Performance:')
print(f'Average Loss: {test_loss:.4f}')
print(f'Accuracy: {correct_test}/{total_test} ({test_acc:.2f}%)')
print(f"Evaluation Time: {eval_end_time - eval_start_time:.2f}s")

# --- Save the Model State Dictionary ---
print(f"\nSaving model state dictionary to {MODEL_FILENAME}...")

if USE_TORCH_COMPILE and hasattr(model, '_orig_mod'):
    print("* Saving state_dict from model._orig_mod (original model)...")
    save_state_dict = model._orig_mod.state_dict()
else:
    print("* Saving state_dict from model (uncompiled)...")
    save_state_dict = model.state_dict()

# Save the selected state dictionary
torch.save(save_state_dict, MODEL_FILENAME)
print("Model saved successfully.")
