import roboticstoolbox as rtb
from spatialmath import SE3,SO3
import numpy as np

robot = rtb.models.URDF.Panda()
#R = np.array([[1,0,0],[0,1,0],[0,0,1]])
pose = robot.fkine(np.random.normal(0,3,(7,)))
while -np.arccos(pose.R[2,2])+np.pi/4<0:
    pose = robot.fkine(np.random.normal(0,3,(7,)))
print(pose)
print(np.arccos(pose.R[2,2]))
print(-np.arccos(pose.R[2,2])+np.pi/4>=0)
test=True