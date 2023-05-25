import numpy as np
import roboticstoolbox as rtb
from matplotlib import pyplot as plt


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


	lim = [(-np.pi, np.pi), (-np.pi, np.pi)]
	rob = rtb.models.DH.Planar2()
	#rob.plot(np.reshape(q,(2,)))
	#qout = project_to_constraint(q, constr, lim, rob)
	#rob.plot(qout)

	xs = np.array([0])
	ys = np.array([0])
	deg = np.pi/180
	for i in range(90):
		for j in range(90):
			q = np.array([4*i*deg,4*j*deg])-np.pi
			q = np.reshape(q,(2,1))
			qout = project_to_constraint(q,constr,lim,rob)
			xs = np.vstack([xs,qout[0]])
			ys = np.vstack([ys, qout[1]])
			print(i,',',j)

	plt.scatter(xs,ys)
	print('hello world')
