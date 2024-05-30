import torch
from torch import nn
import torch.nn.functional as F


class Nerf(nn.Module):
    def __init__(self):
        super(Nerf, self).__init__()
        self.linear_pre = nn.Linear(60, 256)
        self.net = nn.ModuleList()
        for i in range(7):
            if i == 4:
                self.net.append(nn.Linear(256 + 60, 256))
            else:
                self.net.append(nn.Linear(256, 256))
        self.alpha = nn.Linear(256, 1)
        self.feature = nn.Linear(256, 256)
        self.view = nn.Linear(280, 128)
        self.RGB = nn.Linear(128, 3)

    #input_pts is x,y,z  view is how to see the x,y,z
    def forward(self, input_pts, input_view):
        #input_pts = [b,60]   input_vies = [b,24]
        h = F.relu(self.linear_pre(input_pts))
        for i, _ in enumerate(self.net):
            h = F.relu(self.net[i](h))
            if i == 3:
                h = torch.cat([input_pts, h], -1)
        #h = [b,256]
        alpha = F.relu(self.alpha(h))  #alpha= [b,1]
        feature = self.feature(h)      #feature = [b,256]

        h = torch.cat([feature, input_view], -1)   #h = [b,280]

        h = F.relu(self.view(h))       #h = [b,128]
        rgb = F.sigmoid(self.RGB(h)) #rgb = [b,3]

        return rgb, alpha



