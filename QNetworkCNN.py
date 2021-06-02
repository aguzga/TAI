import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class QNetworkCNN(nn.Module):


    def __init__(
        self,
        in_channel=4,
        channel=32,
        n_res_block=2,
        n_res_channel=64,
        stride=4,
        action_dim=5
        ):
      
        super(QNetworkCNN, self).__init__()

        self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel, stride)
        self.fc_1 = nn.Linear(320, 512)
        self.fc_2 = nn.Linear(512, action_dim)
      

    def forward(self, inp):
        
        #inp = inp.view((4, 20, 10))
        x = self.enc.forward(inp)
        print(x.size())
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)

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