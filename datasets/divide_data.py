# import os
# import shutil
# import pandas as pd
#
# input_folder = "data/test"
# labels_csv = "data/train_balanced.csv"
# output_folder = "data/sorted_train"
#
# df = pd.read_csv(labels_csv)
#
#
# for label in df['level'].unique():
#     class_dir = os.path.join(output_folder, f'class_{label}')
#     os.makedirs(class_dir, exist_ok=True)
#
# for _, row in df.iterrows():
#     img_name = row['image'] + ".jpeg"
#     label = row['level']
#     src_path = os.path.join(input_folder, img_name)
#     dst_path = os.path.join(output_folder, f'class_{label}', img_name)
#     if os.path.exists(src_path):
#         shutil.copy2(src_path, dst_path)
#     else:
#         print(f"Nie znaleziono pliku: {img_name}")
#
# print("Obrazy podzielone na klasy.")

import os
import shutil
from sklearn.model_selection import train_test_split

input_dir = "data/sorted_train"
output_base = "data"
splits = ['train', 'val', 'test']
ratios = [0.8, 0.1, 0.1]

for class_folder in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_folder)
    files = os.listdir(class_path)

    train_files, temp_files = train_test_split(files, test_size=(1 - ratios[0]), random_state=42)

    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    split_map = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split, split_files in split_map.items():
        split_class_dir = os.path.join(output_base, split, class_folder)
        os.makedirs(split_class_dir, exist_ok=True)
        for file in split_files:
            shutil.copy2(os.path.join(class_path, file), os.path.join(split_class_dir, file))

print("Podział na train/val/test 80/10/10 gotowy.")