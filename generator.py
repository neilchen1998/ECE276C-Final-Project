#import stuff
from abc import ABC, abstractmethod
from projection import project_to_constraint_scipy as project
import torch
import numpy as np
import roboticstoolbox as rtb
import os
from matplotlib import pyplot as plt
import time
# from two_link import *


HIDDEN = 35
LATENT = 20

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
        self.robot = rtb.models.URDF.Panda()
        def constr2(X):
            ang_diff = -np.arccos(X.R[2,2])+np.pi/10
            return ang_diff
        self.constraints = [constr2]
        self.constr_types = ['ineq']
        self.j_lims = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), \
                       (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
        #self.dataset = np.load('dataset.npy')
        #self.raw = np.load('unprojected.npy')
    
    def generate(self,seed):
        '''
        takes in a random seed as np.array of shape (2,) and returns a sample
        '''
        # if self.constraints[0](self.robot.fkine(seed))>=0:
        #     return seed
        sample = project(seed,self.constraints,self.constr_types,self.j_lims,self.robot)
        #dataset = np.load('dataset.npy')
        #dataset = np.vstack([dataset,sample])
        #raw = np.load('unprojected.npy')
        #raw = np.vstack([raw,seed])
        #np.save('dataset.npy',dataset)
        #np.save('unprojected.npy',raw)
        return sample


##Todo
class VAE_Encoder(torch.nn.Module):
    def __init__(self,num_joints,hidden_size,latent_size):
        super(VAE_Encoder,self).__init__()
        self.linear1 = torch.nn.Linear(num_joints,hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size,1)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(1,hidden_size)
        self.relu3 = torch.nn.ReLU()
        self.linear_mu = torch.nn.Linear(hidden_size,latent_size)
        self.linear_sigma = torch.nn.Linear(hidden_size,latent_size)
    def forward(self,q):
        h = self.relu1(self.linear1(q))
        h = self.relu3(h+self.linear3(self.relu2(self.linear2(h))))
        mu = self.linear_mu(h)
        sigma = torch.exp(self.linear_sigma(h))
        return mu,sigma
##Todo
class VAE_Decoder(torch.nn.Module):
    def __init__(self,num_joints,hidden_size,latent_size):
        super(VAE_Decoder,self).__init__()
        self.linear1 = torch.nn.Linear(latent_size,hidden_size)
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
    def __init__(self,num_joints,hidden_size,latent_size):
        super(VAE_model,self).__init__()
        self.encoder = VAE_Encoder(num_joints,hidden_size,latent_size)
        self.decoder = VAE_Decoder(num_joints,hidden_size,latent_size)
    
    def forward(self,q):
        mu,sigma = self.encoder(q)
        epsilon = torch.randn_like(sigma)
        z = mu+sigma*epsilon
        x = self.decoder(z)
        return x,mu,sigma
    
    def generate(self,z):
        with torch.no_grad():
            out = self.decoder(torch.from_numpy(z).float()).detach().numpy().astype(np.float64)
        return out

class VAE_Generator(Generator):
    """
    Generator which uses VAE to generate samples on the constraint
    """
    def __init__(self):
        self.robot = rtb.models.URDF.Panda()
        def constr2(X):
            ang_diff = -np.arccos(X.R[2,2])+np.pi/10
            return ang_diff
        self.constraints = [constr2]
        self.constr_types = ['ineq']
        self.j_lims = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), \
                       (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
        self.model = VAE_model(num_joints=7,hidden_size=HIDDEN,latent_size=LATENT)
        files = os.listdir(os.getcwd())
        pretrained_file = 'pretrained_panda.tar'
        if pretrained_file in files:
            checkpoint = torch.load(os.path.join(os.getcwd(),pretrained_file))
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.dataset = np.load('dataset.npy')
            self.train(1000,4096)
        self.samples = self.model.generate(np.random.normal(0,3,(3000,LATENT)))
    def generate(self,seed):
        #sample = self.model.generate(seed)
        
        sample = self.samples[seed,:]

        if self.constraints[0](self.robot.fkine(sample))>=0:
            return sample
        projected = project(sample,self.constraints,self.constr_types,self.j_lims,self.robot)
        #projected = sample
        #dataset = np.load('dataset.npy')
        #dataset = np.vstack(dataset,projected)
        #np.save('dataset.npy',dataset)
        #generated = np.load('generated.npy')
        #generated = np.vstack([generated,sample])
        #np.save('generated.npy',generated)
        return projected
    def train(self,num_epochs,samples_per_batch):
        optim = torch.optim.Adam(self.model.parameters(),lr=0.0001,betas=[0.9,0.999999])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,'min',threshold=0.001,min_lr=10**(-10))
        for epoch in range(num_epochs):
            mse_sum = 0
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
            mse = ((batch - batch_hat)**2).sum()/samples_per_batch
            mse_sum+=mse.detach().cpu().item()
            loss = ((batch - batch_hat)**2).sum()/samples_per_batch + (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()/samples_per_batch
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step(loss.item())
            if epoch%10==0:
                print('batch ',i+1,' of ',num_batches,' in epoch ',epoch)
                print('training loss: ',loss.item())
        print('done training')
        print(mse_sum/(num_batches))
        torch.save({'model_state_dict':self.model.state_dict()},os.path.join(os.getcwd(),'pretrained_panda.tar'))
    
if __name__ == '__main__':
    '''data = np.zeros((1,2))
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
    print(data2)'''

    dataset = np.zeros((7,))
    root = os.path.join(os.getcwd(),'export_panda')
    files = os.listdir(root)
    for f in files:
        if 'samples' not in f:
            continue
        file = os.path.join(root,f)
        data = np.load(file)
        dataset = np.vstack([dataset,data])
    dataset = dataset[1:,:]
    np.save('dataset.npy',dataset)
    '''for h in range(2,15,2):
        for l in range(2,h+1,2):
            print('h{}l{}'.format(h,l))
            HIDDEN=h
            LATENT=l
            vae = VAE_Generator()
            gen = Baseline_Generator()
            outs = np.zeros((2,))
            start_t = time.time_ns()
            for i in range(1800):
                if i%100==0:
                    print(i)
                out = vae.generate(np.random.normal(0,1,(LATENT,)))
                #out = gen.generate(np.random.normal(0,1,(2,)))
                outs = np.vstack([outs,out])
            elapsed = (time.time_ns()-start_t)/(10**9)
            #print(elapsed)
            plt.scatter(outs[:,0],outs[:,1])
            plt.title('hidden {} latent {} projected, time={}'.format(h,l,elapsed))
            plt.xlim(-3.5,3.5)
            plt.ylim(-3.5,3.5)
            plt.savefig(os.path.join(os.getcwd(),'temp','h{}l{}p.png'.format(h,l)))
            plt.close()
            #plt.show()'''
    
    
    vae = VAE_Generator()
    #gen = Baseline_Generator()
    start_t = time.time_ns()
    idx = np.random.randint(0, 3000)
    outs = np.zeros((7,))
    for i in range(3000):
        if i%100==0:
            print(i)
        out = vae.generate(idx)
        idx+=1
        idx%=3000
        #out = gen.generate(np.random.normal(0,1,(7,)))
        outs = np.vstack([outs,out])
    elapsed = (time.time_ns()-start_t)/(10**9)
    print(elapsed)
    print(outs[-1])
    robot = rtb.models.URDF.Panda()
    robot.plot(outs[-1], 'pyplot')
    plt.show()
    # plt.scatter(outs[:,0],outs[:,1])
    # plt.title('hidden {} latent {} projected, time={}'.format(HIDDEN,LATENT,elapsed))
    # plt.xlim(-3.5,3.5)
    # plt.ylim(-3.5,3.5)
    # plt.show()
    test=True
