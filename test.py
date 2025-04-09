import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from datasets.dataset import RetinopathyDataset
from models.retfound_mae import get_retfound

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

csv_path = os.path.join('data', 'trainLabels.csv')
root_dir = os.path.join('data', 'train')
test_dataset = RetinopathyDataset(csv_file=csv_path, root_dir=root_dir, transform=transform)
test_dataset = torch.utils.data.Subset(test_dataset, range(50))

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = get_retfound(num_classes=5)
model.load_state_dict(torch.load("retfound_model.pth", map_location=device))
model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on test data: {accuracy:.2f}%")
