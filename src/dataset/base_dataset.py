import  os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .aug import random_augment, just_resize

"""
Dataset
"""
def one_hot_label(num_classes, index_class):
    label = [0 for i in range(num_classes)]
    label[index_class] = 1

    return torch.Tensor(label)


class CustomDataset(Dataset):
    def __init__(self, mode="train", data_folder="", labels={}, num_classes=1000):
        super(CustomDataset, self).__init__()

        assert mode == "train" or mode == "valid", "Mode must be train or valid !"

        self.img_paths = []
        self.mode = mode
        self.num_classes= num_classes
        self.data_folder = data_folder
        self.labels = labels

        self.transform_train = transforms.Compose([
                                        transforms.Lambda(lambda x: random_augment(x)),
                                        transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                             std =[0.229, 0.224, 0.225])
                                        ])
        
        self.transform_valid = transforms.Compose([
                                        transforms.Lambda(lambda x: just_resize(x)),
                                        transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                             std =[0.229, 0.224, 0.225])
                                        ])
        
        for class_name in os.listdir(self.data_folder):
            class_folder = os.path.join(self.data_folder, class_name)
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                self.img_paths.append(image_path)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if self.mode == "train":
            image_tensor = self.transform_train(image)
        else:
            image_tensor = self.transform_valid(image)

        label = img_path.split('/')[-2]
        label_tensor = one_hot_label(self.num_classes, self.labels[label])

        return image_tensor, label_tensor
    
    def __len__(self):
        return len(self.img_paths)

            