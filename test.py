import roboticstoolbox as rtb
from spatialmath import SE3,SO3
import numpy as np
import matplotlib.pyplot as plt

robot = rtb.models.URDF.Panda()
#R = np.array([[1,0,0],[0,1,0],[0,0,1]])
'''joints = np.clip(np.random.normal(0,3,(7,)),-3,3)

pose = robot.fkine(joints)
while -np.arccos(pose.R[2,2])+np.pi/4<0:
    joints = np.clip(np.random.normal(0,3,(7,)),-3,3)
    pose = robot.fkine(joints)
print(pose)
print(-np.arccos(pose.R[2,2])+np.pi/4)
print(-np.arccos(pose.R[2,2])+np.pi/4>=0)
print(joints)'''
#pose = robot.fkine(robot.qr)
#print(robot.fkine(robot.qr).R)
##print(-np.arccos(pose.R[2,2]))
#robot.plot(robot.qr,'pyplot')
pose = SE3(-0.5,0.5,0.7)
joints = robot.ikine_LM(pose,q0 = [1.44426457,  0.75387713, -0.93784086, -1.39904889, -2.53033945,  1.29290135,  0.31331037]).q
robot.plot(joints,'pyplot')
print(joints)
print(-np.arccos(pose.R[2,2])+np.pi/4)
print(robot.fkine(joints))
test=True