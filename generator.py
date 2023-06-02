#import stuff
from abc import ABC, abstractmethod

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
    
    def generate(self,seed):
        '''
        takes in a random seed and returns a sample
        '''
        pass