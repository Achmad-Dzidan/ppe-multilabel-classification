import pandas as pd
import os
import pickle

# Direktori dataset
data_dir = './dataset'
output_file = './dataset/dataset.pkl'

# Baca anotasi
annotations = pd.read_csv(os.path.join(data_dir, 'annotations.csv'))

# Daftar gambar dan label
data = []
for idx, row in annotations.iterrows():
    image_path = os.path.join(data_dir, 'images', row['image_name'])
    if os.path.exists(image_path):
        labels = [row['Glasses'], row['NoGlasses'], row['Helmet'], row['NoHelmet'], row['Mask'], row['NoMask']]
        data.append({'image_path': image_path, 'labels': labels})

     # Simpan ke file pickle
with open(output_file, 'wb') as f:
    pickle.dump(data, f)
print(f'Dataset disimpan ke {output_file}')