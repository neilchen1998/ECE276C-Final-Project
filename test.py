import roboticstoolbox as rtb
from spatialmath import SE3,SO3
import numpy as np
import matplotlib.pyplot as plt

robot = rtb.models.URDF.Panda()
print(robot.fkine(robot.qz))
robot.plot(robot.qz, 'pyplot')
plt.show(-1)