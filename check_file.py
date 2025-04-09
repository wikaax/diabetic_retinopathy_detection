import os
from PIL import Image

root_dir = 'data/train'

for filename in os.listdir(root_dir):
    if filename.endswith('.jpeg'):
        try:
            img_path = os.path.join(root_dir, filename)
            Image.open(img_path)
        except OSError:
            print(f"Uszkodzony plik: {filename}")
