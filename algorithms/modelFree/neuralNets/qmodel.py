import torch as th
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self,nS,nA,hidden_size=[32,64]):
        super(QNet,self).__init__()
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
        # Using Xavier/Glorot initialization for weights and zero initialization for biases
        nn.init.xavier_uniform_(self.fc1.weight)  
        nn.init.xavier_uniform_(self.fc2.weight)  
        nn.init.xavier_uniform_(self.out.weight)  
        nn.init.zeros_(self.fc1.bias)  
        nn.init.zeros_(self.fc2.bias)  
        nn.init.zeros_(self.out.bias) 
