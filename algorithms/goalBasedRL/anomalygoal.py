import torch
import torch.nn as nn
import torch.optim as optim

class AnomalyDetector(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyDetector, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

def reconstruction_score(goals_tensor, model):
    reconstructed = model(goals_tensor)
    return torch.mean((goals_tensor - reconstructed) ** 2, dim=1)  

def log_p_valid(goals, model, temperature=0.1):
    
    score = reconstruction_score(goals, model)
    return -score / temperature 
