#import stuff
from abc import ABC, abstractmethod
from projection import project_to_constraint_scipy as project
import torch
import numpy as np
import roboticstoolbox as rtb

class Generator(ABC):
    """
    Abstract base class for generators
    """
    @abstractmethod
    def __init__(self,constraints,constr_types,joint_lims,robot):
        '''
        initialize
        '''
        pass
    @abstractmethod
    def generate(self,seed):
        '''
        takes in a random seed and returns a sample
        '''
        pass

class Baseline_Generator(Generator):
    """
    Generator which just projects samples to constrained space
    """
    def __init__(self):
        self.robot = rtb.models.DH.Planar2()
        def constr2(X):
            rot = X.angvec()
            ang = (np.pi/2) -rot[0]*np.sign(np.sum(rot[1]))
            ang = abs(ang)
            ang_diff = (np.pi/4)-ang
            return ang_diff
        self.constraints = [constr2]
        self.constr_types = ['ineq']
        self.j_lims = [(-np.pi, np.pi), (-np.pi, np.pi)]
        self.dataset = np.load('dataset.npy')
        self.raw = np.load('unprojected.npy')
    
    def generate(self,seed):
        '''
        takes in a random seed as np.array of shape (2,) and returns a sample
        '''
        sample = project(seed,self.constraints,self.constr_types,self.j_lims,self.robot)
        dataset = np.load('dataset.npy')
        dataset = np.vstack([dataset,sample])
        raw = np.load('unprojected.npy')
        raw = np.vstack([raw,seed])
        np.save('dataset.npy',dataset)
        np.save('unprojected.npy',raw)
        return sample


##Todo
class VAE_Encoder(torch.nn.Module):
    def __init__(self,num_joints,hidden_size):
        super(VAE_Encoder,self).__init__()
        self.linear1 = torch.nn.Linear(num_joints,hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size,1)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(1,hidden_size)
        self.relu3 = torch.nn.ReLU()
        self.linear_mu = torch.nn.Linear(hidden_size,num_joints)
        self.linear_sigma = torch.nn.Linear(hidden_size,num_joints)
    def forward(self,q):
        h = self.relu1(self.linear1(q))
        h = self.relu3(h+self.linear3(self.relu2(self.linear2(h))))
        mu = self.linear_mu(h)
        sigma = torch.exp(self.linear_sigma(h))
        return mu,sigma
##Todo
class VAE_Decoder(torch.nn.Module):
    def __init__(self,num_joints,hidden_size):
        super(VAE_Decoder,self).__init__()
        self.linear1 = torch.nn.Linear(num_joints,hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size,1)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(1,hidden_size)
        self.relu3 = torch.nn.ReLU()
        self.linear_out = torch.nn.Linear(hidden_size,num_joints)
    def forward(self,z):
        h = self.relu1(self.linear1(z))
        h = self.relu3(h+self.linear3(self.relu2(self.linear2(h))))
        x = self.linear_out(h)
        return x
    
class VAE_model(torch.nn.Module):
    def __init__(self,num_joints,hidden_size):
        super(VAE_model,self).__init__()
        self.encoder = VAE_Encoder(num_joints,hidden_size)
        self.decoder = VAE_Decoder(num_joints,hidden_size)
    
    def forward(self,q):
        mu,sigma = self.encoder(q)
        epsilon = torch.randn_like(sigma)
        z = mu+sigma*epsilon
        x = self.decoder(z)
        return x,mu,sigma
    
    def generate(self,z):
        with torch.no_grad():
            out = self.decoder(z).detach().numpy()
        return out

class VAE_Generator(Generator):
    """
    Generator which uses VAE to generate samples on the constraint
    """
    def __init__(self):
        self.robot = rtb.models.DH.Planar2()
        def constr2(X):
            rot = X.angvec()
            ang = (np.pi/2) -rot[0]*np.sign(np.sum(rot[1]))
            ang = abs(ang)
            ang_diff = (np.pi/4)-ang
            return ang_diff
        self.constraints = [constr2]
        self.constr_types = ['ineq']
        self.j_lims = [(-np.pi, np.pi), (-np.pi, np.pi)]
        self.model = VAE_model(num_joints=2,hidden_size=3)
        self.dataset = np.load('dataset.npy')
        self.raw = np.load('generated.npy')
        self.train(100,16)
    def generate(self,seed):
        sample = self.model.generate(seed)
        projected = project(sample,self.constraints,self.constr_types,self.j_lims,self.robot)
        dataset = np.load('dataset.npy')
        dataset = np.vstack(dataset,projected)
        np.save('dataset.npy',dataset)
        generated = np.load('generated.npy')
        generated = np.vstack([generated,sample])
        np.save('generated.npy',generated)
        return projected
    def train(self,num_epochs,samples_per_batch):
        optim = torch.optim.Adam(self.model.parameters(),lr=0.0005,weight_decay=0.000001)
        for epoch in range(num_epochs):
            self.model.train()
            num_batches = self.dataset.shape[0]//samples_per_batch
            for i in range(num_batches):
                batch = torch.from_numpy(self.dataset[i*samples_per_batch:(i+1)*samples_per_batch,:]).float()
                batch_hat,mu,sigma = self.model(batch)
                loss = ((batch - batch_hat)**2).sum()/samples_per_batch + (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()/samples_per_batch
                optim.zero_grad()
                loss.backward()
                optim.step()
            batch = torch.from_numpy(self.dataset[i*samples_per_batch:,:]).float()
            batch_hat,mu,sigma = self.model(batch)
            loss = ((batch - batch_hat)**2).sum()/samples_per_batch + (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()/samples_per_batch
            optim.zero_grad()
            loss.backward()
            optim.step()
            print('batch ',i+1,' of ',num_batches,' in epoch ',epoch)
            print('training loss: ',loss.item())
        print('done training')
    
if __name__ == '__main__':
    data = np.zeros((1,2))
    np.save('dataset.npy',data)
    np.save('unprojected.npy',data)
    np.save('generated.npy',data)
    gen = Baseline_Generator()
    for i in range(1000):
        if i %10==0:
            print(i)
        item = gen.generate(np.random.normal(0,3,(2,)))
    data = np.load('dataset.npy')
    data = data[1:,:]
    np.save('dataset.npy',data)
    data2 = np.load('unprojected.npy')
    data2 = data2[1:,:]
    np.save('unprojected.npy',data2)
    print(data)
    from matplotlib import pyplot as plt
    plt.scatter(data[:,0],data[:,1])
    plt.show()
    print(data2)
    #vae = VAE_Generator()