import os
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from skimage import io
import torchvision.transforms as transforms



#This is why functional > OOP
class ConvertTargets():
    def __init__(self, targets):
        self.targets = targets
        
    def convert_targets(targets):
        num_targets = []
        
        for y in targets:
            
         if y == 'COVID':
             num_targets.append(0)
         elif y == 'Viral':
             num_targets.append(1)
         elif y == 'Lung_':
             num_targets.append(2)
         else:
             num_targets.append(3)
        
        return num_targets


class XrayDataset(Dataset):
    def __init__(self, root_dir, csv_file, transforms = None):
            self.root_dir = root_dir
            self.df = pd.read_csv(csv_file)
            self.transform = transforms
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        img_path = os.path.join(self.root_dir, self.df.iloc[index,1])
        img = io.imread(img_path)
        img  = Image.open(img_path)
        transform = transforms.Grayscale()
        img = transform(img)
        
        y_label = self.df.iloc[index,1][0:5]
        
        if self.transform:
            img = self.transform(img)
        return img, y_label
        