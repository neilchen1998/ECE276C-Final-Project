#!/usr/bin/env python

# source: https://ompl.kavrakilab.org/OptimalPlanning_8py_source.html

import sys
from math import sqrt
import argparse
import numpy as np

try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

from helper import *

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def allocateObjective(si, objectiveType):

    """Select the desired objective function

    Keyword arguments:
    si -- the system info
    objectiveType -- the type of the objective function: PathLength, PathClearance, ThresholdPathLength, WeightedLengthAndClearanceCombo
    """

    if objectiveType.lower() == "pathclearance":
        return getClearanceObjective(si)
    elif objectiveType.lower() == "pathlength":
        return getPathLengthObjective(si)
    elif objectiveType.lower() == "thresholdpathlength":
        return getThresholdPathLengthObj(si)
    elif objectiveType.lower() == "weightedlengthandclearancecombo":
        return getBalancedObjective1(si)
    else:
        ou.OMPL_ERROR("Optimization-objective is not implemented in allocation function.")


def allocatePlanner(si, plannerType):

    """Select the desired planner type

    Keyword arguments:
    si -- the system info
    plannerType -- the type of the planner: RRTstar, BFMTstar, BITstar, FMTstar, InformedRRTstar, PRMstar, SORRTstar
    """

    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        return og.BITstar(si)
    elif plannerType.lower() == "fmtstar":
        return og.FMT(si)
    elif plannerType.lower() == "informedrrtstar":
        return og.InformedRRTstar(si)
    elif plannerType.lower() == "prm":
        return og.PRM(si)
    elif plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    elif plannerType.lower() == 'rrt':
        return og.RRT(si)
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")

def allocMyValidStateSampler(si):

    """Returns an instance of my sampler

    Keyword arguments:
    si -- the system info
    """
    return MyBaselineStateSampler(si)

def plan(runTime, plannerType, objectiveType, s: tuple = (0.0, 0.0), g: tuple = (1.0, 1.0), fname: str ='export'):

    """Plan the path using the specific type of planner

    Keyword arguments:
    runTime -- the limit runtime
    plannerType -- the type of the planner: RRTstar, BFMTstar, BITstar, FMTstar, InformedRRTstar, PRMstar, SORRTstar
    objectiveType -- the type of the objective function: PathLength, PathClearance, ThresholdPathLength, WeightedLengthAndClearanceCombo
    s -- the start position
    g -- the goal position
    fname -- the name of the output file
    """

    # construct the state space of the robot
    space = ob.RealVectorStateSpace(2)

    # set the bound of the space to be in [-np.pi, np.pi]
    space.setBounds(-np.pi, np.pi)

    # construct a state information instance
    si = ob.SpaceInformation(space)

    # set the checker to check which states are valid
    validityChecker = ValidityChecker(si)
    si.setStateValidityChecker(validityChecker)

    # set the sampler
    si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(allocMyValidStateSampler))

    si.setup()

    # export the planner data
    planner_data = ob.PlannerData(si)

    # set the starting point of the robot to be the bottom-left
    start = ob.State(space)
    start[0] = s[0]
    start[1] = s[1]

    # set the ending point of the robot to be the top-right
    goal = ob.State(space)
    goal[0] = g[0]
    goal[1] = g[1]

    # create a problem instance
    pdef = ob.ProblemDefinition(si)

    # set the start and the goal states
    pdef.setStartAndGoalStates(start, goal)

    # create the according optimization objective 
    pdef.setOptimizationObjective(allocateObjective(si, objectiveType))

    # crete the according optimal planner instance
    optimizingPlanner = allocatePlanner(si, plannerType)
    optimizingPlanner.setProblemDefinition(pdef)
    optimizingPlanner.setup()

    def myFilter(a, b) -> bool:

        '''Rejects connections if the distance between two node are larger than a threshold

        Keyword arguments:
        a -- vertex a
        b -- vertex b
        '''

        # update the planner
        optimizingPlanner.getPlannerData(planner_data)

        # check the distance between two nodes
        if (optimizingPlanner.distanceFunction(a, b) > np.pi/8):
            return False
        else:
            return True

    optimizingPlanner.setConnectionFilter(og.ConnectionFilter(myFilter))

    # attempt to solve the problem within the given runtime
    solved = optimizingPlanner.solve(runTime)

    if solved:
         
        # Output the solution info
        print('{0} found solution of path length {1:.4f} with an optimization objective value of {2:.4f}'.format( \
            optimizingPlanner.getName(), \
            pdef.getSolutionPath().length(), \
            pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value()))
         
        # get the solution path
        path = pdef.getSolutionPath().printAsMatrix()

        # convert it to numpy array type
        res = np.array(list(map(lambda x: np.fromstring(x, dtype=np.float32, sep=' '), path.split('\n'))))[:-2]
        
        # save it to npy file
        np.save('{}.npy'.format(fname), res)
        
        # check if a filename is provided
        if fname:
            
            # export the solution path
            with open('{}.txt'.format(fname), 'w+') as outFile:
                outFile.write(pdef.getSolutionPath().printAsMatrix())

        # update the planner data
        # optimizingPlanner.getPlannerData(planner_data)
        # for i in range(planner_data.numVertices()):
        #     # TODO: export the state to of numpy format
        #     x, y = planner_data.getVertex(i).getState()[0]

    
    else:
        print("No solution found.")

if __name__ == "__main__":

    runTime = 30
    planner = 'PRM'
    objective = 'PathLength'
    s, g = (np.pi/4, 0.0), (0.75*np.pi, -np.pi/2)
    fname = 'path-PRM'

    plan(runTime, planner, objective, s, g, fname=fname)
