# FINAL PROJECT AA STEFANO COSTANZO


import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
#import checkNNGradients as checkNNG
from collections import Counter

def main():
	""" Upload the valores array"""
	# Import the values
	valores = read_csv("spam.csv", header=None).to_numpy()
	
	X = valores[1:, 1]
	y = valores[1:, 0]
	m = np.shape(X)[0]
	y = np.reshape(y, (m,1))
	
	# Spam = 0, ham = 1
	for i in range(m):
		if(y[i] == 'spam'):
			y[i] = 0
		else:
			y[i] = 1

	# PREPARATION PART: GENERATE MATRIX IN INPUT FOR THE ALGORITHM, all the supplementary variables: perturb, y_onehot: with 2 columns, y datasets for train and test
	# APPLY NEURONAL ALGORITHM: FIND THETA1 AND THETA2
	# Use a little perturbation in the initial theta default variables
	perturb = 0.12
	y_onehot = y_expand(y, 2)
		
	# DIVIDE TO TRAIN AND TEST PORCION
	perc = int(m * 0.8)
	y_train = y_onehot[:perc, :]
	y_test = y_onehot[perc:, :]
	y_final = y[perc:, :]
	
	# Fix num_ocultas = 100
	num_ocultas = 100
	
	# Apply k-validation function to find the best number of num_entradas (num X) for the system
	Matrix_entradas = np.arange(1000, 9000, 2000)
	k = 5; 
	
	
	# THE FOLLOWING PARTS ARE COMMENTED BECAUSE THEY HAVE TAKES A LOT OF TIME TO EXECUTE ALL THE ITERATIONS:
	# VALIDATION FUNCTION:
	# num_entradas = k_validation(X, y, k, num_ocultas, Matrix_entradas)
	
	# PRINT THE CHARTS:
	# chart3d(X, y)
	chart2d(X, y)
	
	"""
	num_entradas = 5000
	
	# EXECUTE NEURONAL ALGORITHM FOR THE BEST SOLUTION (take as measurement the accuracy)
	X_words = find_occurance(X, num_entradas)
	X_train = X_words[:perc, :]
	X_test = X_words[perc:, :]
	
        # Initialize Theta[i] random in (-perturb, +perturb)
	Theta1_new = np.random.rand(num_ocultas, num_entradas + 1) * (2*perturb) - perturb
	Theta2_new = np.random.rand(2, num_ocultas + 1) * (2*perturb) - perturb
	
	# Put Theta1 and Theta2 in the same linear array 
	params_rn = np.concatenate((np.ravel(Theta1_new), np.ravel(Theta2_new)))
	
	# Execute minimize operation
	params_rn = opt.minimize(fun=backprop, x0=params_rn, args=(num_entradas,num_ocultas,2,X_train,y_train,0), method='TNC', jac=True, options={'maxiter' : 70}).x
	
	# Reshape Theta1, Theta2
	Theta1_new = np.reshape(params_rn[:num_ocultas *(num_entradas + 1)],(num_ocultas, num_entradas + 1));
	Theta2_new = np.reshape(params_rn[num_ocultas *(num_entradas + 1):],(2, num_ocultas + 1));
		
	# Calculate Coste, accuracy, recall and precision
	J = coste_func_reg(Theta1_new, Theta2_new, X_test, y_test, 2, 0)
	accuracy, recall, precision = valuate(Theta1_new, Theta2_new, X_test, y_final)
	accuracy, precision, recall, J = round(accuracy,2), round(precision,2), round(recall,2), round(J,2)
	print("\n[BEST ACCURACY SOLUTION]:")
	print("Entradas:", num_entradas, "ocultas:", num_ocultas, "Accuracy:", accuracy, "precision:", precision, "recall:", recall, "coste:", J)
	print()
	"""
	
def k_validation(X, y, k, num_ocultas, Matrix_entradas):
	"""Function that applies the k-validation test: Divides the dataset to train and test parts (80%-20%): As a result calculate the mean of all 
	the accuracies and return the best solutions of num_entradas """
	# Initial variables
	m = np.shape(X)[0]
	perturb = 0.12
	y_onehot = y_expand(y, 2)
		
	# DIVIDE TO TRAIN AND TEST PORCION FOR y
	perc = int(m * 0.8)
	y_train = y_onehot[:perc, :]
	y_test = y_onehot[perc:, :]
	y_final = y[perc:, :]
	
	# Matrix of all accuracy
	Matrix_accuracy = np.zeros([np.shape(Matrix_entradas)[0], k])
	
	print("K_VALIDATION FUNCTION:")
	print("k:", k, "num_ocultas:", num_ocultas, "\n")
	# Execute for different num_entradas (different x)
	for i in range(np.shape(Matrix_entradas)[0]):
		# DIVIDE TO TRAIN AND TEST PORCION FOR X (respect to the num_entradas)
		num_entradas = Matrix_entradas[i]
		X_words = find_occurance(X, num_entradas)
		X_train = X_words[:perc, :]
		X_test = X_words[perc:, :]
		
		# Execute the k-validation
		for j in range(k):
			start, end = int(perc*(j)/k), int(perc*(j+1)/k)
			X_k_val = X_train[start:end, :]
			X_k_train = np.delete(X_train, range(start,end))
			
			# Initialize Theta[i] random in (-perturb, +perturb)
			Theta1_new = np.random.rand(num_ocultas, num_entradas + 1) * (2*perturb) - perturb
			Theta2_new = np.random.rand(2, num_ocultas + 1) * (2*perturb) - perturb
	
			# Put Theta1 and Theta2 in the same linear array 
			params_rn = np.concatenate((np.ravel(Theta1_new), np.ravel(Theta2_new)))
	
			# Execute minimize operation
			params_rn = opt.minimize(fun=backprop, x0=params_rn, args=(num_entradas,num_ocultas,2,X_train,y_train,0), method='TNC', jac=True, options={'maxiter' : 50}).x
	
			# Reshape Theta1, Theta2
			Theta1_new = np.reshape(params_rn[:num_ocultas *(num_entradas + 1)],(num_ocultas, num_entradas + 1));
			Theta2_new = np.reshape(params_rn[num_ocultas *(num_entradas + 1):],(2, num_ocultas + 1));
	
			# Find the accuracy
			accuracy, recall, precision = valuate(Theta1_new, Theta2_new, X_test, y_final)
			Matrix_accuracy[i,j] = accuracy
			print("num_entradas:", num_entradas, "cicle:", j, "-> accuracy:", accuracy) 
	
	# FIND THE BEST SOLUTION:
	index = np.argmax(np.mean(Matrix_accuracy, axis=1))
	num_entradas = Matrix_entradas[index]
	return num_entradas


def find_occurance(X, num_entradas):
	""" Function that return matrix of occurrence of the num_entradas most common words
	    most_occur (num_entradas): Select most frequent words                
	    Matrix X_words (m x num_entradas): says if the word is present inside the strings x if X_words[i,j] = 1 the word j of most_occur is present inside the mail i    
	"""          	

	m = np.shape(X)[0]
	str = " ".join(tuple(X))

	# split() returns list of all the words in the string
	split_it = str.split()
	
	# Pass the split_it list to instance of Counter class.
	Counters_found = Counter(split_it)
	#print(Counters)

	# most_common() produces k frequently encountered
	# input values and their respective counts.
	most_occur = np.array(Counters_found.most_common(num_entradas))[:, 0]

	X_words = np.zeros([m, num_entradas])
	for i in range(m):
		for j in range(num_entradas):
			if most_occur[j] in X[i]:
				X_words[i, j] = 1
				
	return X_words	
	
def valuate(Theta1, Theta2, X, y):
	"""Returns the accuracy, recall and precision of the system. (X, y) are test dataset"""
	m = X.shape[0]	
	A1, A2, H = forward_prop(Theta1, Theta2, X)
	Tp, Tn, Fp, Fn = 0, 0, 0, 0
	for i in range(m):	
		# Calculate True positive (Tp), true negative (Tn), false positive (Fp), false negative (Fn)
		prediction = np.argmax(H[i][:])
		if(prediction == y[i]):        
			if(prediction == 1):
				Tp += 1
			else:
				Tn += 1
		else:
			if(prediction == 1):
				Fp += 1
			else:
				Fn += 1
	accuracy = (Tp + Tn) / m * 100
	recall = Tp / (Tp + Fn) * 100
	precision = Tp / (Tp + Fp) * 100
	return accuracy, recall, precision

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
	""" Function of the Neural Network. Returns (Coste, gradient) and needs the dates and shapes of the network"""
	m = X.shape[0]	
	
	# Unroll The input matrix
	Theta1 = np.reshape(params_rn[:num_ocultas *(num_entradas + 1)],(num_ocultas,num_entradas + 1));
	Theta2 = np.reshape(params_rn[num_ocultas *(num_entradas+ + 1):],(num_etiquetas,num_ocultas + 1));
	
	# Initialize Delta and Delta2 at zero with some shapes of Theta1 and Theta2
	Delta1 = np.zeros([num_ocultas,num_entradas + 1])
	Delta2 = np.zeros([num_etiquetas,num_ocultas + 1])
	
	# Calculate the coste
	coste = coste_func_reg(Theta1, Theta2, X, y, num_etiquetas, reg)
	
	# Forward propagation
	A1, A2, H = forward_prop(Theta1, Theta2, X)

	# Backword propagation 
	for i in range(m):
		a1i = A1[i, :]
		a2i = A2[i, :]
		hi = H[i, :]
		yi = y[i]
		d3i = hi -yi
		d2i = np.dot(Theta2.T, d3i) * (a2i *(1 - a2i))

		# Calculate the gradient
		Delta1 = Delta1 + np.dot(d2i[1:, np.newaxis], a1i[np.newaxis,:])
		Delta2 = Delta2 + np.dot(d3i[:, np.newaxis], a2i[np.newaxis,:])
		

	
	# Finalize the calculatuon of gradient and the regolarization
	Delta1 = (1/m) * Delta1 + np.hstack([np.zeros([num_ocultas,1]), (reg/m) * Theta1[:, 1:]]) 
	Delta2 = (1/m) * Delta2 + np.hstack([np.zeros([num_etiquetas,1]), (reg/m) * Theta2[:, 1:]])
	
	gradiente = np.concatenate((np.ravel(Delta1), np.ravel(Delta2)))
	return (coste, list(gradiente))
	
def forward_prop(Theta1, Theta2, X):
	""" Forward propagation. Return A1, A2 and H (The final results of the Neuronal Network)"""
	m = X.shape[0]
	A1 = np.hstack([np.ones([m, 1]), X])
	Z2 = np.dot(A1, Theta1.T) 
	A2 = np.hstack([np.ones([m, 1]),sigmoide(Z2)]) 
	Z3 = np.dot(A2, Theta2.T) 
	H = sigmoide(Z3)
	return A1, A2, H


def sigmoide(x):
	return 1/(1+np.exp(-x))

def coste_func(Theta1, Theta2, X, y, num_etiquetas):
	""" Coste function"""
	m = X.shape[0]
	A1, A2, H = forward_prop(Theta1, Theta2, X)
	
	#y_onehot = y_expand(y, num_etiquetas)
	Coste = 0
	for i in range(num_etiquetas):
		log1 = np.log(H[:,i])  #
		log2 = np.log(1 - H[:,i])
		Coste += (-1 / m) * (np.matmul(np.transpose(log1), y[:,i]) + np.matmul(np.transpose(log2), 1 - y[:,i]))
	return Coste
		
def coste_func_reg(Theta1, Theta2, X, y, num_etiquetas, Lambda):
	""" Coste function with regularization"""
	m = X.shape[0]
	Coste = coste_func(Theta1, Theta2, X, y, num_etiquetas) + Lambda / (2 * m) * (np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))
	return Coste
	
def y_expand(y, num_etiquetas):
	""" Expanding function for y : (m, 1) -> (m, num_etiquetas) """
	m = np.shape(y)[0]
	y_onehot = np.zeros((m, num_etiquetas))
	for i in range(m):
		y_onehot[i][y[i][0]] = 1
	return y_onehot
	
def chart3d(X, y):
	""" Print the 4 charts (x = num_entradas, y = num_ocultas, z = (case 1: Coste, case 2: Accuracy, case 3: Recall, case 4: Precision)"""
	
	# Initialize variables, perturb, y 
	m = X.shape[0]	
	perturb = 0.12
	perc = int(m * 0.7)
	y_onehot = y_expand(y, 2)
	y_train = y_onehot[:perc, :]
	y_test = y_onehot[perc:, :]
	y_final = y[perc:, :]
	
	# Range of tests 1000 < X < 7000, y = [50, 100, 300]
	num_entradas = np.arange(1000, 8000, 2000)
	num_ocultas = np.array([50, 100, 300])
	
	# Meshgrid
	M1, M2 = np.meshgrid(num_entradas, num_ocultas)
	
	# Initialize empty matrix J, accuracy, recall, precision
	J = np.zeros(np.shape(M1))
	accuracy = np.zeros(np.shape(M1))
	recall = np.zeros(np.shape(M1))
	precision = np.zeros(np.shape(M1))
	
	#
	for i, j in np.ndindex(M1.shape):
		# DIVIDE TO TRAIN AND TEST PORCION
		X_words = find_occurance(X, M1[i,j])
		X_train = X_words[:perc, :]
		X_test = X_words[perc:, :]
		
		# Initialize Theta[i] random in (-perturb, +perturb)
		Theta1_new = np.random.rand(M2[i,j], M1[i,j] + 1) * (2*perturb) - perturb
		Theta2_new = np.random.rand(2, M2[i,j] + 1) * (2*perturb) - perturb
		
		# Put Theta1 and Theta2 in the same linear array 
		params_rn = np.concatenate((np.ravel(Theta1_new), np.ravel(Theta2_new)))
		
		# Execute minimize operation
		params_rn = opt.minimize(fun=backprop, x0=params_rn, args=(M1[i,j],M2[i,j],2,X_train,y_train,0), method='TNC', jac=True, options={'maxiter' : 70}).x
	
		# Reshape Theta1, Theta2
		Theta1_new = np.reshape(params_rn[:M2[i,j] *(M1[i,j] + 1)],(M2[i,j], M1[i,j] + 1));
		Theta2_new = np.reshape(params_rn[M2[i,j] *(M1[i,j] + 1):],(2, M2[i,j] + 1));
		
		# Calculate with function valuate and coste_func_reg the measurement
		accuracy[i][j], recall[i][j], precision[i][j] = valuate(Theta1_new, Theta2_new, X_test, y_final)
		J[i][j] = coste_func_reg(Theta1_new, Theta2_new, X_test, y_test, 2, 0)
		accuracy[i][j], precision[i][j], recall[i][j], J[i][j] = round(accuracy[i][j],2), round(precision[i][j],2), round(recall[i][j],2), round(J[i][j],2)
		print("entradas:", M1[i,j], "ocultas:", M2[i,j], "Accuracy:", accuracy[i][j], "precision:", precision[i][j], "recall: ", recall[i][j], "coste:", J[i][j])

	# Find the best solutions for every measurement and print on the console
	max_J, max_acc, max_prec, max_rec = np.argmax(J), np.argmax(accuracy), np.argmax(precision), np.argmax(recall)
	i, j = max_J // np.shape(J)[1], max_J % np.shape(J)[1]
	print("\n[BEST J] entradas:", M1[i,j], "ocultas:", M2[i,j], "Accuracy:", accuracy[i][j], "precision:", precision[i][j], "recall: ", recall[i][j], "coste:", J[i][j])
	i, j = max_acc // np.shape(J)[1], max_acc % np.shape(J)[1]
	print("\n[BEST accuracy] entradas:", M1[i,j], "ocultas:", M2[i,j], "Accuracy:", accuracy[i][j], "precision:", precision[i][j], "recall: ", recall[i][j], "coste:", J[i][j])
	i, j = max_rec // np.shape(J)[1], max_rec % np.shape(J)[1]
	print("\n[BEST recall] entradas:", M1[i,j], "ocultas:", M2[i,j], "Accuracy:", accuracy[i][j], "precision:", precision[i][j], "recall: ", recall[i][j], "coste:", J[i][j])
	i, j = max_prec // np.shape(J)[1], max_prec % np.shape(J)[1]
	print("\n[BEST precision] entradas:", M1[i,j], "ocultas:", M2[i,j], "Accuracy:", accuracy[i][j], "precision:", precision[i][j], "recall: ", recall[i][j], "coste:", J[i][j])
	
	
	# Plot 3-d charts for the 4 case-study
	fig = plt.figure()
	ax1 = fig.add_subplot(2, 2, 1, projection='3d')
	ax2 = fig.add_subplot(2, 2, 2, projection='3d')
	ax3 = fig.add_subplot(2, 2, 3, projection='3d')
	ax4 = fig.add_subplot(2, 2, 4, projection='3d')
	
	ax1.set_title('J')
	surf = ax1.plot_surface(M1, M2, J, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax2.set_title('Accuracy')
	surf = ax2.plot_surface(M1, M2, accuracy, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax3.set_title('Recall')
	surf = ax3.plot_surface(M1, M2, recall, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax4.set_title('Precision')
	surf = ax4.plot_surface(M1, M2, precision, cmap=cm.coolwarm, linewidth=0, antialiased=False)	
	plt.show()
	
def chart2d(X, y):
	# Initialize variables, perturb, y 
	m = X.shape[0]	
	perturb = 0.12
	perc = int(m * 0.7)
	y_onehot = y_expand(y, 2)
	y_train = y_onehot[:perc, :]
	y_test = y_onehot[perc:, :]
	y_final = y[perc:, :]
	
	# Range of tests 1000 < X < 7000
	num_entradas = np.array([ 500, 1000, 3000, 5000, 7000, 10000, 15000])
	num_ocultas = 2	

	
	# Initialize empty matrix J, accuracy, recall, precision
	res = np.zeros([np.shape(num_entradas)[0], 4])
	
	for i in range(np.shape(num_entradas)[0]):
		# DIVIDE TO TRAIN AND TEST PORCION
		X_words = find_occurance(X, num_entradas[i])
		X_train = X_words[:perc, :]
		X_test = X_words[perc:, :]
		
		# Initialize Theta[i] random in (-perturb, +perturb)
		Theta1_new = np.random.rand(num_ocultas, num_entradas[i] + 1) * (2*perturb) - perturb
		Theta2_new = np.random.rand(2, num_ocultas + 1) * (2*perturb) - perturb
		
		# Put Theta1 and Theta2 in the same linear array 
		params_rn = np.concatenate((np.ravel(Theta1_new), np.ravel(Theta2_new)))
		
		# Execute minimize operation
		params_rn = opt.minimize(fun=backprop, x0=params_rn, args=(num_entradas[i],num_ocultas,2,X_train,y_train,0), method='TNC', jac=True, options={'maxiter' : 50}).x
	
		# Reshape Theta1, Theta2
		Theta1_new = np.reshape(params_rn[:num_ocultas *(num_entradas[i] + 1)],(num_ocultas, num_entradas[i] + 1));
		Theta2_new = np.reshape(params_rn[num_ocultas *(num_entradas[i] + 1):],(2, num_ocultas + 1));
		
		# Calculate with function valuate and coste_func_reg the measurement
		res[i][1], res[i][2], res[i][3] = valuate(Theta1_new, Theta2_new, X_test, y_final)
		res[i][0] = coste_func_reg(Theta1_new, Theta2_new, X_test, y_test, 2, 0)
		res[i][1], res[i][2], res[i][3], res[i][0] = round(res[i][1],2), round(res[i][2],2), round(res[i][3],2), round(res[i][0],2)

	fig = plt.figure()
	plt.plot(num_entradas, res[:, 1], c="red", label="Accuracy")
	plt.plot(num_entradas, res[:, 2], c ="green", label="Precision")
	plt.plot(num_entradas, res[:, 3], c="blue", label="Recall")
	plt.legend()
	plt.savefig("part2.png")

main()
