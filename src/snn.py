import torch
import torch.nn as nn
import snntorch as snn

class SNN(nn.Module):
    def __init__(self, layerDims, beta):
        super().__init__()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.fc1 = nn.Linear(layerDims[0], layerDims[1])
        self.lif1 = snn.Leaky(beta=beta) 
        self.fc2 = nn.Linear( layerDims[1],  layerDims[2])
        self.lif2 = snn.Leaky(beta=beta)

        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()

    def forward(self, x):
        
        cur1 = self.fc1(x)
        spk1, self.mem1 = self.lif1(cur1, self.mem1)
        cur2 = self.fc2(spk1)
        spk2, self.mem2 = self.lif2(cur2, self.mem2)

        return spk2
        
        

