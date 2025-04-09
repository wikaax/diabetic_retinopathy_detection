import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.dataset import RetinopathyDataset
from models.retfound_mae import get_retfound
from models.medvit import get_medvit
from models.swin_transformer import get_swin
from models.mae import get_mae
from datasets.dataset import collate_fn

def get_model(model_name, num_classes=5):
    if model_name == "retfound":
        return get_retfound(num_classes)
    elif model_name == "medvit":
        return get_medvit(num_classes)
    elif model_name == "swin":
        return get_swin(num_classes)
    elif model_name == "mae":
        return get_mae(num_classes)
    else:
        raise ValueError("Nieznany model!")

# if gpu available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

csv_path = os.path.join('data', 'trainLabels.csv')
df = pd.read_csv(csv_path)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['level'], random_state=42)

train_df.to_csv("data/train_split.csv", index=False)
val_df.to_csv("data/val_split.csv", index=False)

train_dataset = RetinopathyDataset(csv_file="data/train_split.csv", root_dir="data/train", transform=transform)
val_dataset = RetinopathyDataset(csv_file="data/val_split.csv", root_dir="data/train", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model_name = "retfound"  # model selection
model = get_model(model_name).to(device)

# loss definition
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        if images is None or labels is None:
            print("Pominięto próbkę z brakującymi danymi")
            continue

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")

torch.save(model.state_dict(), f"{model_name}_model.pth")
print("Trening zakończony!")
