import warnings
warnings.filterwarnings('ignore')
import torch
import os, time, datetime
from config import get_config
from torch.utils.data import DataLoader
from src.dataset import CustomDataset
from src.models import create_model
from cls_utils import *

def get_dict_label_mapping(folder_data):
        count = 0
        res = {}
        for class_name in os.listdir(folder_data):
            res[class_name] = count
            count += 1        
        return res

def print_verbose(verbose=True, *args, **kwargs): print(*args) if verbose else None

class Runner:
    def __init__(self, cfg_all = None, dict_label_mapping={}):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cfg_all = cfg_all
        self.model_name = self.cfg_all['model']['model_name']
        self.pretrained = self.cfg_all['model']['pretrained']
        self.save_path = self.cfg_all['model']['save_path']
        os.makedirs(self.save_path, exist_ok=True)
        
        self.dict_label_mapping = dict_label_mapping
        self.epoch = self.cfg_all['training']['epoch']
        self.num_classes = self.cfg_all['data']['num_classes']

        self.model = create_model(self.model_name, self.num_classes)
        if self.pretrained:
            self.model = self.model.load(self.pretrained)

        self.model = self.model.to(self.device)
        self.train_dataset = CustomDataset(mode="train", 
                                           data_folder=self.cfg_all['data']['root_train'], 
                                           labels=self.dict_label_mapping,
                                           num_classes=self.num_classes)
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, num_workers=4, shuffle=True)

        self.valid_dataset = CustomDataset(mode="valid",
                                           data_folder=self.cfg_all['data']['root_valid'], 
                                           labels=self.dict_label_mapping,
                                           num_classes=self.num_classes)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=16, num_workers=4)


        self.loss_fn = intit_loss()
        self.optimizer = init_optimizeer(model=self.model)
        self.scheduler = init_scheduler(optimizer=self.optimizer)


    def train_step(self, epoch):
        self.model.train()
        
        train_correct = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)

            loss.backward()
            self.optimizer.step()

            pred = torch.argmax(output, dim=1)
            target = torch.argmax(target, dim=1)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % 100 == 0:
                iter_num = batch_idx * len(data)
                total_data = len(self.train_loader.dataset)
                iter_num = str(iter_num).zfill(len(str(total_data)))
                total_percent = 100. * batch_idx / len(self.train_loader)
                print_verbose(True, f'Train Epoch {epoch + 1}: [{iter_num}/{total_data} ({total_percent:2.0f}%)] | Loss: {loss.item():.10f} | LR: {self.optimizer.param_groups[0]["lr"]:.10f}')
                
        train_accuracy = 100. * train_correct / len(self.train_loader.dataset)
        return train_accuracy


    def valid_step(self):
        self.model.eval()
        valid_correct = 0
        
        for (data, target) in self.valid_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            pred = torch.argmax(output, dim=1)
            target = torch.argmax(target, dim=1)
            valid_correct += pred.eq(target.view_as(pred)).sum().item()

        valid_accuracy = 100. * valid_correct / len(self.valid_loader.dataset)
        return valid_accuracy
    
    
    def train(self):
        max_valid_accuracy = 0.0
        for epoch in range(self.epoch):
            tik = time.time()
            train_accuracy = self.train_step(epoch)
            valid_accuracy = self.valid_step()
            print_verbose(True, f'Training accuracy: {train_accuracy}%')
            print_verbose(True, f'Validating accuracy: {valid_accuracy}%')
            print()
            
            self.image_classifier = self.model
            
            if valid_accuracy >= max_valid_accuracy:
                max_valid_accuracy = valid_accuracy
                self.image_classifier.save(os.path.join(self.save_path, 'best.pth'))
                print("Saving best weight in {} at epoch_{}".format(self.save_path, epoch+1))

            if os.path.exists(os.path.join(self.save_path, f'epoch_{epoch - 2}.pth')):
                os.remove(os.path.join(self.save_path, f'epoch_{epoch - 2}.pth'))
            self.image_classifier.save(os.path.join(self.save_path, f'epoch_{epoch}.pth'))

            self.scheduler.step()

            tok = time.time()
            runtime = tok - tik
            eta = int(runtime * (self.epoch - epoch - 1))
            eta = str(datetime.timedelta(seconds=eta))
            print_verbose(True, f'Runing time: Epoch {epoch + 1}: {str(datetime.timedelta(seconds=int(runtime)))} | ETA: {eta}')
            print()
            print()



if __name__ == '__main__':
    cfg_all = get_config()
    dict_label_mapping = get_dict_label_mapping(cfg_all['data']['root_train'])
    print("Check dict mapping labels:", dict_label_mapping)

    runner = Runner(cfg_all=cfg_all, dict_label_mapping=dict_label_mapping)
    runner.train()
