import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_name = row['image'].lower() + ".jpeg"
        img_path = os.path.join(self.root_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except OSError:
            print(f"Nie można otworzyć obrazu: {img_path}. Pomijam.")
            return None, None

        label = int(row['level'])

        if self.transform:
            image = self.transform(image)

        return image, label


def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]

    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])

    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)

    return images, labels
