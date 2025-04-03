import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from datasets.dataset import RetinopathyDataset
from models.retfound_mae import get_retfound
from models.medvit import get_medvit
from models.swin_transformer import get_swin
from models.mae import get_mae

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
root_dir = os.path.join('data', 'train')

train_dataset = RetinopathyDataset(csv_file=csv_path, root_dir=root_dir, transform=transform)
train_dataset = torch.utils.data.Subset(train_dataset, range(10))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model_name = "retfound"  # model selection
model = get_model(model_name).to(device)

# loss definition
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10

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

print("Trening zakończony!")
torch.save(model.state_dict(), f"{model_name}_model.pth")
print(torch.cuda.is_available())