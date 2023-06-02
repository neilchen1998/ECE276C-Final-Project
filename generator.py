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
        # i need to hardcode these
        self.robot = rtb.models.DH.Planar2()
        def constr2(X):
            x = X.t
            return 0.5-x[0]
        self.constraints = [constr2]
        self.constr_types = ['ineq']
        self.j_lims = [(-np.pi, np.pi), (-np.pi, np.pi)]
    
    def generate(self,seed):
        '''
        takes in a random seed as np.array of shape (2,) and returns a sample
        '''
        return project(seed,self.constraints,self.constr_types,self.j_lims,self.robot)


##Todo
class VAE_Encoder(torch.nn.Module):
    def __init__(self):
        super(VAE_Encoder,self).__init__()
    def forward(self,x):
        return x
##Todo
class VAE_Decoder(torch.nn.Module):
    def __init__(self):
        super(VAE_Decoder,self).__init__()
    def forward(self,x):
        return x

class VAE_Generator(Generator):
    """
    Generator which uses VAE to generate samples on the constraint
    """
    def __init__(self,constraints,constr_types,joint_lims,robot):
        '''
        constraints: a list of constraint functions that take a cartesian space point as input
        constr_types: list of types 'eq' for equality, 'ineq' for inequality
        joint_lims: the joint limits of the robot as a list of tuples [(min,max),...]
        robot: the robot used for fkine
        '''
        self.robot = robot
        self.constraints = constraints
        self.constr_types = constr_types
        self.j_lims = joint_lims
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
        self.data = np.empty((2,))
        #generate first random points
        self.train()
    def generate(self,seed):
        pass
    def train(self):
        pass
    def save_dataset(self):
        return self.data
    
