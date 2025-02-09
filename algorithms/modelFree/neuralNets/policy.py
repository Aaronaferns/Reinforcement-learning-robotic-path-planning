import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self,nS,nA,hidden_size=[32,64]):
        super(Policy,self).__init__()
        self.fc1=nn.Linear(nS,hidden_size[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0],hidden_size[1])
        self.out = nn.Linear(hidden_size[1],nA)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.out(x)
        return x

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')  
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')  
        nn.init.kaiming_uniform_(self.out.weight, nonlinearity='relu')  
        nn.init.zeros_(self.fc1.bias)  
        nn.init.zeros_(self.fc2.bias)  
        nn.init.zeros_(self.out.bias) 

