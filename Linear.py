import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Dataset import XrayDataset, ConvertTargets


class NN(nn.Module):
  def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fit1 = nn.Linear(input_size, 50)
        self.fit2 = nn.Linear(50, num_classes)

  def forward(self, X):
    layer_1 = F.relu(self.fit1(X))
    layer_2 = self.fit2(layer_1)

    return layer_2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#299*299
input_size = 89401
num_classes = 4
batch_size = 64
learning_rate = 0.001
epochs = 2

dataset = XrayDataset(root_dir = 'Normal_COVID_Lung_Viral', csv_file = 'Normal_COVID_Lung_Viral.metadata.csv', transforms= transforms.ToTensor())
training, test = torch.utils.data.random_split(dataset, [0.7, 0.3])
loading_dataset = DataLoader(dataset=training, batch_size=batch_size, shuffle=True)
loading_testdataset = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)


model = NN(input_size = input_size, num_classes = num_classes).to(device)

criterion = nn.CrossEntropyLoss()

opimtimzer = optim.Adam(model.parameters(), lr = learning_rate)


for epochs in range(epochs):
  for batch_idx, (data, targets) in enumerate(loading_dataset):

    # data = data.to(device = device)
    
    # targets = targets.to(device = device)
    
    
    data = data.view(data.shape[0], -1)

    scores = model(data)
    
    num_targets = ConvertTargets.convert_targets(targets)
    num_targets = torch.Tensor(num_targets).long()
    
    
    loss = criterion(scores, num_targets)

    opimtimzer.zero_grad()
    loss.backward()

    opimtimzer.step()


def check_acc(loader, model):



    num_samples = 0
    num_correct = 0
    model.eval()

    with torch.no_grad():
      for x,y in loader:
          
        # x = x.to(device = device)
        # y = y.to(device = device)
        x = x.view(x.shape[0], -1)
        
        targets_0 = ConvertTargets.convert_targets(targets)
        targets_0 = torch.Tensor(targets_0)

        scores = model(x)
        _, predictions = scores.max(1)
        num_correct = num_correct + (predictions == targets_0).sum()
        num_samples = num_samples + predictions.size(0)
      acc = (float(num_correct)/ float(num_samples))*100
      print(f'Acc: {acc}:2f')
    model.train()


check_acc(loading_dataset, model)
check_acc(loading_testdataset, model)