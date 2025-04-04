from math import floor, log10
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Dataloader import XrayDataset, ConvertTargets



class CNN(nn.Module):
    #if you add more layers be mindful of the in and out channels and nn.Linear
    #can try avg pooling
    def __init__(self, in_channels = 1, num_classes=4):
          super(CNN, self).__init__()
          self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
          self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
          self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
          self.ft1 = nn.Linear(16*74*74, num_classes)
          
    def forward(self, X):
       #NB selu does batch norm (bn) so is a quick fix until we can sort bn
       layer_1 = F.selu(self.conv1(X))
       layer_2 = self.max_pool(layer_1)
       layer_3 = F.selu(self.conv2(layer_2))
       layer_4 = self.max_pool(layer_3)
       
       x = layer_4.reshape(layer_4.shape[0], -1)
       print(x.shape, 'forward')
       x = self.ft1(x)
       #if x.shape[0]==64:
       return x
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hps
#299*299
input_size = 89401
num_classes = 4
batch_size = 64
learning_rate = 0.001
epochs = 5

#add your transforms here
img_transforms = transforms.Compose([
    transforms.ToTensor()])

dataset = XrayDataset(root_dir = 'Normal_COVID_Lung_Viral', csv_file = 'Normal_COVID_Lung_Viral.metadata.csv', transforms= transforms.ToTensor())
training, test = torch.utils.data.random_split(dataset, [0.7, 0.3])
loading_dataset = DataLoader(dataset=training, batch_size=batch_size, shuffle=True)
loading_testdataset = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)


model = CNN().to(device)

criterion = nn.CrossEntropyLoss()

opimtimzer = optim.Adam(model.parameters(), lr = learning_rate)


for epochs in range(epochs):
  for batch_idx, (data, targets) in enumerate(loading_dataset):
      
    #uncomment when using cuda  

    # data = data.to(device = device)
    # targets = targets.to(device = device)
    
    scores = model(data)
    
    num_targets = ConvertTargets.convert_targets(targets)
    num_targets = torch.Tensor(num_targets).long()
    
    
    loss = criterion(scores, num_targets)

    opimtimzer.zero_grad()
    loss.backward()

    opimtimzer.step()


def check_acc(loader, model, train_set=True):

    num_samples = 0
    num_correct = 0
    model.eval()

    with torch.no_grad():
        
      for x,y in loader:
          
        #uncomment when using cuda  
          
        # x = x.to(device = device)
        # y = y.to(device = device)
       
        
        targets_0 = ConvertTargets.convert_targets(y)
        targets_0 = torch.Tensor(targets_0)

        scores = model(x)
        _, predictions = scores.max(1)
        
        num_correct = num_correct + (predictions == targets_0).sum()
        num_samples = num_samples + predictions.size(0)
      acc = (float(num_correct)/ float(num_samples))*100
      acc  = round(acc, -int(floor(log10(abs(acc)))) + (3 - 1))
      if train_set:
          print(f'Training Acc: {acc}')
      else:
          print(f'Test Acc: {acc}')
    model.train()


check_acc(loading_dataset, model)
check_acc(loading_testdataset, model, False)