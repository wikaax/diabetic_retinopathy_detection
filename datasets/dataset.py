import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os


class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # filter images so that csv = dir
        self.valid_labels = self._filter_valid_images()

    def _filter_valid_images(self):
        valid_images = []
        for idx, row in self.labels.iterrows():
            img_name_from_csv = row['image'].lower()
            img_path = os.path.join(self.root_dir, img_name_from_csv + ".jpeg")
            if os.path.exists(img_path):
                valid_images.append(row)
        return pd.DataFrame(valid_images)

    def __len__(self):
        return len(self.valid_labels)

    def __getitem__(self, idx):
        row = self.valid_labels.iloc[idx]
        img_name_from_csv = row['image'].lower()
        img_path = os.path.join(self.root_dir, img_name_from_csv + ".jpeg")

        image = Image.open(img_path).convert("RGB")
        label = row['level']

        if self.transform:
            image = self.transform(image)

        return image, label


# image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

csv_path = os.path.join('data', 'trainLabels.csv')
root_dir = os.path.join('data', 'train')

train_dataset = RetinopathyDataset(csv_file=csv_path, root_dir=root_dir, transform=transform)


def collate_fn(batch):
    # filter none from batch
    batch = [item for item in batch if item[0] is not None]

    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])

    images, labels = zip(*batch)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    return images, labels


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

for images, labels in train_loader:
    if images.size(0) == 0:
        continue

    print(images.shape)  # torch.Size([32, 3, 224, 224])
    print(labels)
    break

# files in train dir but not in csv
missing_files = []
for idx, row in train_dataset.valid_labels.iterrows():
    img_name_from_csv = row['image'].lower()
    img_path = os.path.join(root_dir, img_name_from_csv + ".jpeg")
    if not os.path.exists(img_path):
        missing_files.append(img_name_from_csv)

print(f"Brakujące pliki (z CSV, ale nie ma ich w katalogu): {missing_files}")
