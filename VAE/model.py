import torch
from torch import nn
import torch.nn.functional as F

# Input img -> Hidden dim -> mean,std -> reparameterise -> decode -> output
class VariationalAutoEncoder(nn.Module):
    def __init__(self,input_dim, h_dim = 200, z_dim = 20):
        super().__init__()
        
        #encoder
        self.img_2hid = nn.Linear(input_dim,h_dim)
        self.hid_2mu = nn.Linear(h_dim,z_dim)
        self.hid_2sigma = nn.Linear(h_dim,z_dim)

        #Latent space to be standard Gaussian

        #decoder
        self.z_2hid = nn.Linear(z_dim,h_dim)
        self.hid_2img = nn.Linear(h_dim,input_dim)

        self.relu = nn.ReLU()

    def encode(self,x):
        #q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu,sigma = self.hid_2mu(h) , self.hid_2sigma(h)

        return mu,sigma
    
    def decode(self,z):
        #p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        #return normalised value of each pixel (0-1)
        return torch.sigmoid(self.hid_2img(h)) 
    
        

    def forward(self,x):
        mu,sigma = self.encode(x)
        epsilon = torch.randn_like(sigma) #gaussian
        z_reparametrized = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma

if __name__ == "__main__":
    x = torch.randn(4,28*28)
    vae = VariationalAutoEncoder(input_dim = 784)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)
