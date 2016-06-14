from math import log
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def ReadData(filename):
	dataX = []
	dataY = []
	errorY = []

	file = open(filename, 'r')
	with file as f:
		for line in f:
			x, y, ey = line.split('\t')
			#x, y = line.split()
			dataX.append(float(x))
			dataY.append(float(y))
			errorY.append(float(ey))

	file.close()

	return dataX, dataY, errorY

def combineDataSets(data, abs_data_sets, rel_data_sets):
	#All data is merged into one matrix
	mergedDataX = []
	mergedDataY = []
	mergedDataeY = []
	
	for i in range(abs_data_sets):
		mergedDataX += data['absX{0}'.format(i)]
		mergedDataY += data['absy{0}'.format(i)]
		mergedDataeY += data['absey{0}'.format(i)]

	for i in range(rel_data_sets):
		mergedDataX += data['relX{0}'.format(i)]
		mergedDataY += data['rely{0}'.format(i)]
		mergedDataeY += data['reley{0}'.format(i)]		

	return mergedDataX, mergedDataY, mergedDataeY

def createMatrixX(dataX, dataeY, dataY, data, rel_data_sets, x0):
	#Create a Matrix of zero
	matrix_X = [ [0 for x in range(4+rel_data_sets)] for y in range(len(dataX)) ]
	#This is format for matrix_X
	#(x-x0)^2	1	x 	x0(2x-x0), x^2	rel_1	rel_2	rel_3 ...
	#See the paper for more details

	for i in range(len(dataX)):
		if (dataX[i] < x0):
			matrix_X[i][0] = (dataY[i]/dataeY[i])*(log(dataX[i]) - log(x0))**2
			matrix_X[i][3] = (dataY[i]/dataeY[i])*log(x0)*(2*log(dataX[i]) - log(x0))
		else:
			matrix_X[i][3] = (dataY[i]/dataeY[i])*log(dataX[i])**2
		matrix_X[i][1] = (dataY[i]/dataeY[i])*1
		matrix_X[i][2] = (dataY[i]/dataeY[i])*log(dataX[i])
		for rel in range(rel_data_sets):
			if dataX[i] in data['relX{0}'.format(rel)]:
				matrix_X[i][rel + 4] = (dataY[i]/dataeY[i])*1
				break

	return matrix_X

def createMatrixY(dataY, dataeY):
	#matrix_Y is a vector
	matrix_Y = [0 for y in range(len(dataY))]

	for i in range(len(dataY)):
		matrix_Y[i] = (dataY[i]/dataeY[i])*log(dataY[i])
	return matrix_Y

#Define the chisquare function
def ChiSquared(X, Y, theta):
	chi_squared1 = (X*theta - Y.transpose())
	chi_squared2 = chi_squared1.transpose()*chi_squared1
	return chi_squared2

def derChiSquaredWRTx0(dataX, dataY, dataeY, x0, theta, data, rel_data_sets):
	#The derivative of chi squared wrt theta. This is needed for gradient descent.
	der_theta = 0
	
	for i in range(len(dataX)):
		scaling_factor = 0
		if (dataX[i] < x0):
			for rel in range(rel_data_sets):
				if dataX[i] in data['relX{0}'.format(rel)]:
					scaling_factor = theta.item(4+rel, 0)
					break
			x_l = np.matrix([(log(dataX[i]) - log(x0))**2, 1, log(dataX[i]), log(x0)*(2*log(dataX[i]) - log(x0))])
			der_theta += -2*( log(dataY[i]) - x_l*theta[0:4,:] - scaling_factor )*2*((log(dataX[i]) - log(x0)))*( theta.item(0,0) - theta.item(3,0) )
	return der_theta	

def evaluateTheta(X, Y):
	XT = X.transpose()
	theta = (np.linalg.inv(XT*X)*XT*(Y.transpose()))
	return theta

def ChivsE0(dataX, dataY, dataeY, data, rel_data_sets):
	#this function creates a file containg value of E0 and chisquares. E0 starts at 80 keV.	
	target = open('ChivsE0.dat', 'w')
	target.truncate()
	for x0 in range(80, 1000):
		getX = createMatrixX(dataX, dataeY, dataY, data, rel_data_sets, x0)
		getY = createMatrixY(dataY, dataeY)
		#print (getY)
		X = np.matrix(getX)
		Y = np.matrix(getY)
		theta = evaluateTheta(X, Y)
		chisquare = ChiSquared(X, Y, theta)
		target.write(str(x0) + '\t' + str(chisquare.item(0,0)))
		target.write("\n")
	target.close()


def plot(theta, x0, mergedDataX, mergedDataY, data, rel_data_sets, covMat, t):
	#A plot function to help visualize how good the fit is and what the confidence interval looks like
	x = np.linspace(50,3600,10000) #this creates 10000 points between 50 and 3600
	y = []
	dataX = []
	dataY = []
	#upper confidence level
	conf_u = []
	#lower confidence level
	conf_d = []

	for i in range(len(mergedDataX)):
		scaling_factor = 0
		for rel in range(rel_data_sets):
			if mergedDataX[i] in data['relX{0}'.format(rel)]:
				scaling_factor = theta.item(4+rel, 0)
				break
		dataX.append(mergedDataX[i])
		dataY.append(mergedDataY[i]/exp(scaling_factor))

	for i in range(len(x)):
		if x[i] < x0:
			x_l = np.matrix([(log(x[i]) - log(x0))**2, 1, log(x[i]), log(x0)*(2*log(x[i]) - log(x0))])
			var_xl = np.matrix([(log(x[i]) - log(x0))**2, 1, log(x[i]), log(x0)*(2*log(x[i]) - log(x0)), 
				2*((log(x[i]) - log(x0)))*( theta.item(0,0) - theta.item(3,0) )])
			answer = x_l*theta[0:4,:]
			answer = exp(answer.item(0,0))
			y.append(answer)
			
			answer2 = ( var_xl*covMat[0:5,0:5]*var_xl.transpose() )

			answer2 = exp(t*answer2.item(0,0)**0.5)
			conf_u.append(answer*answer2)
			conf_d.append(answer/answer2)
		else:
			x_h = np.matrix([0, 1, log(x[i]), log(x[i])**2])
			answer = x_h*theta[0:4,:]
			answer = exp(answer.item(0,0))
			y.append(answer)
			answer2 = ( x_h*covMat[0:4,0:4]*x_h.transpose() )
			answer2 = exp(t*answer2.item(0,0)**0.5)
			conf_u.append(answer*answer2)
			conf_d.append(answer/answer2)

	

	plt.axis([50, 4000, 0.0008, 0.015])
	plt.loglog(x,y)
	plt.loglog(dataX, dataY, 'ro')
	plt.loglog(x, conf_u)
	plt.loglog(x, conf_d)
	plt.show()


def GradientDescent(dataX, dataY, dataeY, init_x0, data, rel_data_sets):
	precision = 0.00005
	gamma = 0.1
	x_new = init_x0 + 3*precision
	x_old = init_x0
	while abs(x_new - x_old) > precision:
		getX = createMatrixX(dataX, dataeY, dataY, data, rel_data_sets, x_old)
		getY = createMatrixY(dataY, dataeY)
		X = np.matrix(getX)
		Y = np.matrix(getY)
		theta = evaluateTheta(X, Y)
		der_chi = derChiSquaredWRTx0(dataX, dataY, dataeY, x_old, theta, data, rel_data_sets)
		x_old = x_new
		x_new = x_old - gamma*der_chi
	return x_new

def createMatrixX_withx0(dataX, dataeY, dataY, data, rel_data_sets, x0, theta):
	#Create a Matrix of zero
	matrix_X = [ [0 for x in range(5+rel_data_sets)] for y in range(len(dataX)) ]
	#This is format for matrix_X
	#(x-x0)^2	1	x 	x0(2x-x0), x^2	rel_1	rel_2	rel_3

	for i in range(len(dataX)):
		if (dataX[i] < x0):
			matrix_X[i][0] = (dataY[i]/dataeY[i])*(log(dataX[i]) - log(x0))**2
			matrix_X[i][3] = (dataY[i]/dataeY[i])*log(x0)*(2*log(dataX[i]) - log(x0))
			matrix_X[i][4] = (dataY[i]/dataeY[i])*2*((log(dataX[i]) - log(x0)))*( theta.item(0,0) - theta.item(3,0) )
		else:
			matrix_X[i][3] = (dataY[i]/dataeY[i])*log(dataX[i])**2
		matrix_X[i][1] = (dataY[i]/dataeY[i])*1
		matrix_X[i][2] = (dataY[i]/dataeY[i])*log(dataX[i])
		for rel in range(rel_data_sets):
			if dataX[i] in data['relX{0}'.format(rel)]:
				matrix_X[i][rel + 5] = (dataY[i]/dataeY[i])*1
				break

	return matrix_X

def main():
	###Read the absolute and relative efficiency data sets
	data = {}

	abs_data_sets = int(input('How many absolute efficiency data sets: '))

	for i in range(abs_data_sets):
		filename = input('Input the name of the absolute efficiency file: ')
		dataX, dataY, errorY = ReadData(filename)
		#print(dataX)
		data['absX{0}'.format(i)] = dataX
		data['absy{0}'.format(i)] = dataY
		data['absey{0}'.format(i)] = errorY

	rel_data_sets = int(input('How many relative efficiency data sets: '))

	for i in range(rel_data_sets):
		filename = input('Input the name of the relative efficiency file: ')
		dataX, dataY, errorY = ReadData(filename)
		#print(dataX)
		data['relX{0}'.format(i)] = dataX
		data['rely{0}'.format(i)] = dataY
		data['reley{0}'.format(i)] = errorY

	x0 = float(input('Enter an estimate for the intersection point in keV: '))
	mergedDataX, mergedDataY, mergedDataeY = combineDataSets(data, abs_data_sets, rel_data_sets)
	
	ChivsE0(mergedDataX, mergedDataY, mergedDataeY, data, rel_data_sets)

	x0 = GradientDescent(mergedDataX, mergedDataY, mergedDataeY, x0, data, rel_data_sets)
	getX = createMatrixX(mergedDataX, mergedDataeY, mergedDataY, data, rel_data_sets, x0)
	getY = createMatrixY(mergedDataY, mergedDataeY)
	X = np.matrix(getX)
	Y = np.matrix(getY)
	theta = evaluateTheta(X, Y)
	chisquare = ChiSquared(X,Y,theta)

	print('\n')
	print('E_0 = ', x0.item(0,0))
	#Outputs the covariance matrix
	for i in range(len(theta)):
		if i < 4:
			print('alpha_{',i+1,'} = ',  theta.item(i,0))
		else:
			print('rel_{',i-3,'} = ', theta.item(i,0))
	
	chisquare_min = chisquare.item(0,0)/(len(mergedDataX) - 5 - rel_data_sets)
	print('chisquare_min = ', chisquare_min)
	#print(mergedDataY)
	covMat = createMatrixX_withx0(mergedDataX, mergedDataeY, mergedDataY, data, rel_data_sets, x0, theta)
	covMat = np.matrix(covMat)
	#print(covMat.transpose()*covMat)
	covMat = np.linalg.inv(covMat.transpose()*covMat)
	print('covariance matrix = \n')
	for i in range(len(covMat)):
		for j in range(len(covMat)):
			print('%.9f \t' %covMat.item(i,j), end = '')
		print('\n')
	#students t-value
	students_t = (stats.t.ppf(1-0.025, len(mergedDataX) - 5 - rel_data_sets)) 
	plot(theta, x0, mergedDataX, mergedDataY, data, rel_data_sets, covMat*chisquare_min, students_t)


main()