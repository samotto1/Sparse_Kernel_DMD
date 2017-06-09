#####################################
#
# Kernel DMD Testing 
# See example in Dynamic Mode Decomp: Theory and Applications 
#
#####################################

#####################################
# Imports: 
import numpy
from math import *
import pylab
import sys
import matplotlib.colors
import scipy.linalg
import scipy.io as sio
import modred as mr
import matplotlib.pyplot as plt
from cmath import sin, cos, exp, pi, log, polar, rect, phase, sqrt
from future.builtins import range
import scipy.ndimage.filters
import time 
#####################################
t = time.time()

# Hyperparameters:

gamma = 0.1

n_modes = 500		# Number of POD modes to keep 

# Polynomial parameters: 
kernel_power = 5
kern_const = 1
# RBF parameter: 
sigma = 0.3

do_rbf = 0
do_sparse = 1

print('Sigma is: ')
print sigma
print('Gamma is: ')
print gamma
print 




#####################################
# Dependencies: 

def scale_snapshots(X):
	# Scales data matrix by average norm of snapshots 
	avg_sum = 0
	for i in range(len(X[0,:])):
		avg_sum = avg_sum + numpy.linalg.norm(X[:,i])
	avg = avg_sum / len(X[0,:])

	Xscale = numpy.zeros(numpy.shape(X))
	for i in range(len(X[0,:])):
		Xscale[:,i] = X[:,i]/avg
	return Xscale

def compute_kernel_test(X1, Y1):
	# The simplest kernel test. 
	Ahat = numpy.dot(Y1.T,X1)
	Ghat = numpy.dot(X1.T,X1)

	return Ghat, Ahat


def compute_kernel_matrices(X, Y):
	# Choose: f(x,z) = (1 + zTx)^20 (for now) 
	Ghat=numpy.zeros((len(X[0,:]),len(X[0,:])))
	Ahat=numpy.zeros((len(X[0,:]),len(X[0,:])))

	for i in range(len(X[0,:])):
		for j in range(len(X[0,:])):
			Ahat[i,j] = numpy.power((kern_const + numpy.dot(X[:,j].T,Y[:,i])),kernel_power)
			Ghat[i,j] = numpy.power((kern_const + numpy.dot(X[:,j].T,X[:,i])),kernel_power)
	return Ghat, Ahat


def compute_rbf_kernel_matrices(X, Y):
	# f(x,z) = exp(-1/(2sigma) ||x-z||2^2)

	# sigma defined above 

	Ghat=numpy.zeros((len(X[0,:]),len(X[0,:])))
	Ahat=numpy.zeros((len(X[0,:]),len(X[0,:])))
	for i in range(len(X[0,:])):
		for j in range(len(X[0,:])):
			Ahat[i,j] = numpy.exp( -1.0/(2.0*numpy.power(sigma,2))*numpy.power( numpy.linalg.norm(X[:,j] - Y[:,i]),2))
			Ghat[i,j] = numpy.exp( -1.0/(2.0*numpy.power(sigma,2))*numpy.power( numpy.linalg.norm(X[:,j] - X[:,i]),2))
	return Ghat, Ahat






def compute_Khat(w_red, V_red, Ahat):
	# Utilizes original Khat from Matthew William's paper (2015) 
	SigPseu = numpy.diag(1.0/numpy.sqrt(w_red))
	Khat = numpy.dot(numpy.dot(numpy.dot(numpy.dot(SigPseu, V_red.T), Ahat), V_red), SigPseu)
	return Khat, SigPseu



def compute_Khat_sparse(w_red, V_red, Ahat):
	# Utilizes original Khat from Matthew William's paper (2015) 
	# Sig = numpy.diag(numpy.sqrt(w_red))
	SigPseu = numpy.diag(1.0/numpy.sqrt(w_red))

	# gamma modified above! 

	Khat=numpy.zeros((n_modes,n_modes))
	for i in range(n_modes):
		for j in range(n_modes):
			sj = numpy.sqrt(w_red[j])
			si = numpy.sqrt(w_red[i])
			y = (numpy.dot(numpy.dot(V_red[:,i].T, Ahat), V_red[:,j]))

			if(y > gamma/(2.0*si*sj)):
				Khat[i,j] = (y - gamma/(2.0*si*sj))/(si*sj)
			elif(y < -1.0*gamma/(2.0*si*sj)):
				Khat[i,j] = (y + gamma/(2.0*si*sj))/(si*sj)
			else:
				Khat[i,j] = 0.0

	return Khat, SigPseu





# Functions for plotting unit circle in complex: 
def scale_vector(scale, vector):
	result = [0]*len(vector)
	for i in range(len(result)):
		result[i] = scale * vector[i]
	return result
def real_vector(vector):
	return map(lambda x: -1.0*x.real, vector)
def imag_vector(vector):
	return map(lambda x: x.imag, vector)
def plot_complex_eigenvalues(v_in):
	# Plot complex eigenvalues: 
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(numpy.real(v_in), numpy.imag(v_in), picker=True)
	ax.set_title('Eigenvalues of A')
	ax.set_xlabel('Real part of eigenvalues')
	ax.set_ylabel('Imag part of eigenvalues')
	# Generate numbers around the complex unit circle.
	N = 128
	theta = scale_vector(2*pi/N, range(N))
	exp_theta = map(lambda x: exp(1j * x), theta)
	real_part = real_vector(exp_theta)
	imag_part = imag_vector(exp_theta)
	pylab.plot(real_part,imag_part)
	fig.canvas.callbacks.connect('pick_event', on_pick)
	pylab.show()	

def on_pick(event):
        ind = event.ind
	index = numpy.take(ind, 0)
	real_eig = numpy.take(numpy.real(numpy.take(wK, ind)),0)
	imag_eig = numpy.take(numpy.imag(numpy.take(wK, ind)),0)
        print('Eigenvalue Index: ' + repr(index))
	print('Value: ' + repr(numpy.take(numpy.take(wK, ind),0)))

	imaglogeigv = numpy.imag(numpy.log(numpy.take(numpy.take(wK, ind),0)))
	per = float(2.0*pi/imaglogeigv*27.2753/365)

        print('Approx. period in years: ' + repr(per))
	print

	# Mode plot: 
	pylab.figure()
	pylab.subplot(2, 1, 1)
	dmd_m= find_K_mode(index)
	#pylab.plot(numpy.real(dmd_m[:50]), label="Real part")
	#pylab.plot(numpy.imag(dmd_m[:50]), label="Imag part")

	pylab.plot(numpy.linspace(90,-90,num=50), numpy.real(dmd_m[:50]), label="Real part") #, numpy.linspace(0,49,num=50)
	pylab.plot(numpy.linspace(90,-90,num=50), numpy.imag(dmd_m[:50]), label="Imag part")
	pylab.xlim(-90,90)
	pylab.xlabel('Sine(latitude)')
	pylab.ylabel('Amplitude')
	pylab.title('Mode:')
	pylab.legend()

	# Progressing eigenvector plot: 
	pylab.subplot(2, 1, 2)

	# Multiplying dmd_m by powers of mu (no eigenfunction used here) 
	ans_=numpy.zeros((50,300))
	for i in range(300):
		ans_[:,i] = numpy.real(numpy.power(numpy.take(numpy.take(wK, ind),0),i)*dmd_m[:50])

	pylab.imshow(ans_, extent=[0,300,-90,90])
	pylab.colorbar()

	pylab.show(block=False)

def find_K_mode(index):
	# Original method for finding modes. Matthew Williams' paper (2015) 
	dmd_mode = numpy.dot(numpy.dot(numpy.dot(X1, V_red), SigPseu) , numpy.conj(Vleft[:,index]))
	return dmd_mode

def find_K_function(index):
	# Using right eigenvectors to find eigenfunctions
	# initial condition index is start_val 
	start_val = 125
	dmd_fn = numpy.dot(numpy.dot(numpy.dot(Ghat[start_val,:], V_red), SigPseu), Vright[:,index])
	return dmd_fn


#####################################




# -----------------------------------------------------
# Load the data and store in X: 
# Data source: https://solarscience.msfc.nasa.gov/greenwch/bflydata.txt
mat_contents = sio.loadmat('sunspot_data.mat')  # MatLab file of butterfly diagram data
X = mat_contents['data_v']			# Data matrix (50x1907) for analysis.
X_true = mat_contents['data_v']			# A copy of the exact data for comparison later. 
# -----------------------------------------------------

N_samples = X.shape[1]
s_dim = X.shape[0];

#X = scipy.ndimage.filters.gaussian_filter(X,2)

# form Hankel Matrix
overlaps=200;
H=numpy.zeros((s_dim*overlaps,N_samples-overlaps))
for i in range(0,N_samples-overlaps):
	for j in range(0,overlaps):
		H[s_dim*(j):(s_dim*(j+1)),i]=X[:,(i+j)]

X_scale = scale_snapshots(H)

#X_scale = scale_snapshots(X)

# -----------------------------------------------------
# (Standard) Kernel DMD: 

# 1) Choose kernel function: 



# 2) Create matrices Phi(X)*Phi(X) and Phi(Y)*Phi(X) using kernel function: 
X1 = X_scale[:,:-1]
Y1 = X_scale[:,1:]

# Choose kernel function: 
if(do_rbf == False):
	Ghat, Ahat = compute_kernel_matrices(X1, Y1)
else:
	Ghat, Ahat = compute_rbf_kernel_matrices(X1, Y1)

# Symmetry check on Ahat: 
if(numpy.allclose(Ahat, Ahat.T)):
	print('Your Ahat is symmetric! ')
	print 


# 3) Compute Sigma and V of Phi(X)*Phi(X): 

w,V = numpy.linalg.eigh(Ghat)	# eigenvalues w, eigenvectors v (columns)

w_red = w[-n_modes:]
V_red = V[:, -n_modes:]

#pylab.semilogy(w_red, '-x')
#pylab.show()

# 4) Compute Khat (after reducing # of modes) 
if(do_sparse==0):
	Khat, SigPseu = compute_Khat(w_red, V_red, Ahat)
else:
	Khat, SigPseu = compute_Khat_sparse(w_red, V_red, Ahat)

# 5) Find left and right eigenvectors of Khat and eigenvalues 
wK,Vleft = scipy.linalg.eig(Khat, left=True, right=False)
wK,Vright = scipy.linalg.eig(Khat, left=False, right=True)

elapsed = time.time() - t
print('Elapsed time in seconds: ')
print elapsed
print 

# -----------------------------------------------------
# Plotting results: 
plot_complex_eigenvalues(wK)



# -----------------------------------------------------









# -----------------------------------------------------
# Modified Kernel DMD (with L1 penalty function): 

# -----------------------------------------------------










