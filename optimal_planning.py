#!/usr/bin/env python

# source: https://ompl.kavrakilab.org/OptimalPlanning_8py_source.html

import sys
from math import sqrt
import argparse

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
    elif plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")

def plan(runTime, plannerType, objectiveType, fname):

    """Plan the path using the specific type of planner

    Keyword arguments:
    runTime -- the limit runtime
    plannerType -- the type of the planner: RRTstar, BFMTstar, BITstar, FMTstar, InformedRRTstar, PRMstar, SORRTstar
    objectiveType -- the type of the objective function: PathLength, PathClearance, ThresholdPathLength, WeightedLengthAndClearanceCombo
    fname -- the name of the output file
    """

    assert(runTime > 0.0, 'the value of runtime should be greater than 0.0')

    # construct the state space of the robot
    space = ob.RealVectorStateSpace(2)

    # set the bound of the space to be in [0, 1]
    space.setBounds(0.0, 1.0)

    # construct a state information instance
    si = ob.SpaceInformation(space)

    # set the checker to check which states are valid
    validityChecker = ValidityChecker(si)
    si.setStateValidityChecker(validityChecker)

    si.setup()

    # set the starting point of the robot to be the bottom-left
    start = ob.State(space)
    start[0] = 0.0
    start[1] = 0.0

    # set the ending point of the robot to be the top-right
    goal = ob.State(space)
    goal[0] = 0.0
    goal[1] = 0.0

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

    # attempt to solve the problem within the given runtime
    solved = optimizingPlanner.solve(runTime)

    if solved:
         
         # Output the length of the path found
         print('{0} found solution of path length {1:.4f} with an optimization ' \
             'objective value of {2:.4f}'.format( \
             optimizingPlanner.getName(), \
             pdef.getSolutionPath().length(), \
             pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value()))
    
    else:
        print("No solution found.")

if __name__ == "__main__":

    runTime = 30
    planner = 'RRTstar'
    objective = 'PathLength'

    plan(runTime, planner, objective, None)
