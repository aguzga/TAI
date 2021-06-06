import torch
from torch import nn
from torch.nn import functional as F

class QNetworkCNN(nn.Module):


    def __init__(
        self
        ):
      
        super(QNetworkCNN, self).__init__()

        
        #self.fc_0 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        #self.fc_1 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        #self.fc_2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(inplace=True))
        #self.fc_3 = nn.Sequential(nn.Linear(32, 1))

        self.fc_0 = nn.Sequential(nn.Linear(4, 512), nn.ReLU(inplace=True))
        self.fc_1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True))
        self.fc_2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True))
        self.fc_3 = nn.Sequential(nn.Linear(256, 1))


        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
      

    def forward(self, inp):

        x = self.fc_0(inp)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)

        return x
'''
from torchinfo import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QNetworkCNN().to(device)
#print(model)
summary(model, input_size=(512, 4, 20, 10))
#a = torch.zeros((16, 4, 20, 10), device=device)
#model(a)
'''