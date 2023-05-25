import numpy as np
import roboticstoolbox as rtb
from matplotlib import pyplot as plt
import scipy.optimize as opti
import time


def project_to_constraint(q_in, constraint_grad, lims, robot):
	"""
	:param q_in: q-space point to be projected
	:param constraint_grad: gradient of the constraint with respect to x,y,z,theta_x,theta_y,theta_z (function of an SE3 matrix)
	:param lims: joint limits
	:param robot: robot to find jacobians
	:return: the q-space point projected onto the constraint
	"""
	max_iter = 1000
	step_size = 0.1
	qtemp = q_in
	for i in range(max_iter):
		J = robot.jacob0(qtemp)
		x = robot.fkine(np.reshape(qtemp, (2,)))
		grad = constraint_grad(x)
		step = step_size * np.reshape(grad @ J, (2, 1))
		qnext = qtemp - step
		qtemp = qnext
		if np.linalg.norm(step) < 0.0001:
			break
	qtemp = np.reshape(qtemp, (2,))
	return qtemp

def project_to_constraint_scipy(q_in, constraint,consttype, lims, robot):
	"""
	:param q_in: q-space point to be projected, as (2,)
	:param constraint: constraint function on c-space
	:param consttype: 'eq' for equality, 'ineq' for inequality
	:param lims: joint limits as list of tuples
	:param robot: robot for fkine
	:return: the closest point on the constrained space to the given point
	"""
	def loss(x):
		return np.linalg.norm(np.reshape(q_in,x.shape)-x)
	def my_constraint(q):
		x = robot.fkine(np.reshape(q,(2,)))
		return constraint(x)
	consts = [{'type':consttype,'fun':my_constraint},]
	qout = opti.minimize(loss,np.reshape(q_in,(2,)),method='slsqp',bounds=lims,constraints=consts,options={'disp':False, 'ftol':1e-6, 'maxiter':2000})
	return qout.x


if __name__ == '__main__':

	q = np.random.normal(0,1,(2,1))


	# drive to 0
	# def constr(X):
	#	x = X.t
	#	grad = np.reshape(x,(1,3))
	#	grad = np.hstack([grad,np.zeros((1,3))])
	#	return grad

	# drive to left hand plane
	def constr(X):
		x = X.t
		grad = np.zeros((1, 6))
		if x[0] >= 0:
			grad[0, 0] = x[0]
		return grad

	def constr2(X):
		x = X.t
		return -x[0]

	lim = [(-np.pi, np.pi), (-np.pi, np.pi)]
	rob = rtb.models.DH.Planar2()
	rob.plot(np.reshape(q,(2,)))
	#qout = project_to_constraint(q, constr, lim, rob)
	#qout = project_to_constraint_scipy(q,constr2,'ineq',lim,rob)
	#rob.plot(qout)

	fig = plt.figure()
	deg = np.pi/180
	start_t = time.time()
	for i in range(90):
		for j in range(90):
			q = np.array([4*i*deg,4*j*deg])-np.pi
			q = np.reshape(q,(2,1))
			qout = project_to_constraint(q,constr,lim,rob)
	#		qout = project_to_constraint_scipy(q, constr2, 'ineq', lim, rob)
			print(i,',',j)
	print(start_t)
	print(time.time())
	print('hello world')
