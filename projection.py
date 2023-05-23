import numpy as np
import roboticstoolbox as rtb


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
		x = robot.fkine(np.reshape(qtemp,(2,)))
		grad = constraint_grad(x)
		qnext = qtemp-step_size*np.reshape(grad@J,(2,1))
		qtemp = qnext
	return q_out


if __name__ == '__main__':
	print('hello world')
	q = np.zeros((2, 1))+1


	def constr(X):
		x = X.t
		grad = np.reshape(x,(1,3))
		grad = np.hstack([grad,np.zeros((1,3))])
		return grad


	lim = [(-np.pi, np.pi), (-np.pi, np.pi)]
	rob = rtb.models.DH.Planar2()
	rob.plot([0,0])
	qout = project_to_constraint(q, constr, lim, rob)
	rob.plot(qout)
