#!/usr/bin/env python

from ompl import base as ob
from ompl import util as ou
from math import sqrt
import sys

# source https://ompl.kavrakilab.org/StateSampling_8py_source.html

class ValidityChecker(ob.StateValidityChecker):

    def clearance(self, state) -> float:

        """Returns the distance from the position of the given state to the boundary of the circle

        Keyword arguments:
        state -- the given state
        """

        # extract the values of x & y
        x = state[0]
        y = state[1]

        return sqrt((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)) - 0.25
    
    def isValid(self, state) -> bool:

        """Returns whether the position of the given state overlaps the circular obstacle

        Keyword arguments:
        state -- the given state
        """

        return self.clearance(state) > 0.0
    

class ClearanceObjective(ob.StateCostIntegralObjective):

    def __init__(self, si):

        super(ClearanceObjective, self).__init__(si, True)
        self.si_ = si

    def stateCost(self, s):

        """Returns the cost
        Our requirement is to maximize path clearance from obstacles,
        but we want to represent the objective as a path cost
        minimization. Therefore, we set each state's cost to be the
        reciprocal of its clearance, so that as state clearance
        increases, the state cost decreases.

        Keyword arguments:
        s -- the state of the robot
        """

        return ob.Cost(1 / (self.si_.getStateValidityChecker().clearance(s) + sys.float_info.min))
    

def getClearanceObjective(si):

    return ClearanceObjective(si)

def getBalancedObjective1(si):

    lengthObj = ob.PathLengthOptimizationObjective(si)
    clearObj = ClearanceObjective(si)

    opt = ob.MultiOptimizationObjective(si)
    opt.addObjective(lengthObj, 5.0)
    opt.addObjective(clearObj, 1.0)

    return opt

def getPathLengthObjWithCostToGo(si):

    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))

    return obj

def getPathLengthObjective(si):

    return ob.PathLengthOptimizationObjective(si)

def getThresholdPathLengthObj(si):

    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(1.51))
    
    return obj

class MyStateSampler(ob.StateSampler):

    def __init__(self, si):
        super(MyStateSampler, self).__init__(si)
        self.name_ = "my sampler"
        self.rng_ = ou.RNG()

    def sampleUniform(self, state):

        """Override the sample uniform function such that it does not returns points in the top-left quardrant
        A valid states must follow the following constraints:
        0 <= x,y <=1
        if x <= 0.5, then y < 0.5

        Keyword arguments:
        state -- the state of the robot
        """
        
        x = self.rng_.uniformReal(0, 1)
        if x <= 0.5:
            y = self.rng_.uniformReal(0, 0.5)
        else:
            y = self.rng_.uniformReal(0, 1)
        
        # assign the value we generate to state
        state[0] = x
        state[1] = y
