import os
import torch
from datasets.dataset import RetinopathyDataset



dataset = RetinopathyDataset(csv_file="data/trainLabels.csv", root_dir="data/train")
print(torch.cuda.is_available())

