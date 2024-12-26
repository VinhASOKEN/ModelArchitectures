import os
import shutil
from sklearn.model_selection import train_test_split

parent_folder = "/data/disk2/vinhnguyen/ModelArchitectures/Animals"
train_folder = "/data/disk2/vinhnguyen/ModelArchitectures/Animals_splited/train"
valid_folder = "/data/disk2/vinhnguyen/ModelArchitectures/Animals_splited/valid"

os.makedirs(train_folder, exist_ok=True)
os.makedirs(valid_folder, exist_ok=True)

for class_name in os.listdir(parent_folder):
    class_path = os.path.join(parent_folder, class_name)
    if not os.path.isdir(class_path):
        continue
    
    files = os.listdir(class_path)
    files = [os.path.join(class_path, f) for f in files]
    
    train_files, valid_files = train_test_split(files, test_size=0.16, random_state=42)
    
    train_class_folder = os.path.join(train_folder, class_name)
    valid_class_folder = os.path.join(valid_folder, class_name)
    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(valid_class_folder, exist_ok=True)
    
    for f in train_files:
        shutil.copy(f, train_class_folder)
    for f in valid_files:
        shutil.copy(f, valid_class_folder)


shutil.rmtree(parent_folder)