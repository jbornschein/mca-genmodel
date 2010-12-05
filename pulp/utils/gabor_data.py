#!/usr/bin/env python
""" Gabor processing, testing and benchmarking class. 

Statistical tools to test a given gabor data set.
Classifies and fits gabor in data sets.
Produces Gabor test sets.

Why gabors and not Mickey Mouses? Visit the Group-Wiki. 
After all the project is called *gabor*, not Mickey Mouse...

"""

from __future__ import division
from tables import openFile

import numpy as np
from scipy.signal import convolve2d
from numpy.random import uniform, randn, multivariate_normal
from numpy.linalg import eig
from numpy.fft import fft2, ifft2, fftshift

from math import pi
from scipy import inf
from scipy.optimize import fmin_powell

from time import ctime
from os import system, getcwd
from sys import exit, stdout

from mpi4py import MPI

from matplotlib import pyplot
from pulp.utils.autotable import AutoTable
from pulp.utils.parallel import fair_asim_gather, sca_assim_load



#TODO : Separate I/O operations from the functions themselves!
#TODO: Set fontsize globally
 

#Documenting functions, putting structure! adding references
#TODO:To include: Paper references to gabor characterizations (Field, Tolhurst).




################################################################################
def gabor_kern(params, shape):
    """Produces a gabor kernel.  

    *shape*: Tuple: (*height*, *width*)
    *params*: array containing: (*x0*, *y0*, *sigma_x*, *sigma_y*,
    *psi*, *tau*, *k*, *A*). Where *psi* is the wave vector angle
    and *tau* the phase. Consult the group wiki for a detailed 
    description.

    Returns an array of shape *shape* containing the generated gabor.

    """

    # Extract parameters
    x0 = params[0:2]
    #Ppal axes sigma
    sigma_unr = np.array( [[1/params[2]**2, 0], [0, 1/params[3]**2]] )
    psi, tau = params[4:6]
    k0 = params[6]
    A = params[7]
    h, w = shape

    #Rotation matrix used
    R0 = np.array( [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]] )

    #Rotated sigma
    sigma_r = np.dot( np.transpose(R0), np.dot(sigma_unr, R0) )
    k_r = np.dot( np.transpose(R0), np.array([k0, 0]) )

    # Generate Grid, shifted by x0
    gx, gy = np.ogrid[0:h, 0:w]
    gx -= x0[0]
    gy -= x0[1]
    # Generate the exponents and the companion cosine 
    e = sigma_r[1,1]*gx*gx + (sigma_r[0,1]+sigma_r[1,0])*gx*gy + sigma_r[0,0]*gy*gy

    angular = (k_r[1]*gx + k_r[0]*gy) + tau
    gabor_kern = (A*np.exp(-0.5*e))*(np.cos(angular))
    return gabor_kern




def gabor_benchmark(dist_params, gabor_shape, num_basis):
    """ Produce a set of randomly generated gabor functions, according 
    to the given distribution.

    *dist_params* is a list/vector/tuple of (pk0, center_vec, cov_params, 
    bor, A): wave vector's *module*, (2D) *center* of the gaussian 
    distribution to draw from (x and y coord resp), cov_params = (*sigma_x*, 
    *sigma_y*, *angle*) to construct sample covariance matrix from, *borders*
    form the sides, and *amplitud* to use. *gabor_shape* is a 2 element
    list/vector/tuple that specifies the shape of the output. *num_basis* is
    the number of random gabors to be generated. 
    Returns tuple with 1_: 2D of shape (*num_gabor*, *D*) with the random
    gabors flattened in the second dim. 2_: the generative parameters used.
    """

    #Unpack parameters to use:
    pk0, center, cov_params, bor, A = dist_params
    sigma_x, sigma_y, psi = cov_params
    h, w = gabor_shape


    #Preallocate return structures:
    gen_params = np.empty([num_basis, 8])
    gabor_set = np.empty([num_basis, h*w])

    #Assign the fixed parameters *pk0* and *A*.
    gen_params[:, 6] = pk0
    gen_params[:, 7] = A

	#A_We select the centers x0, x1 from a uniform random distribution
    gen_params[:, 0]  = uniform(bor, h - bor, num_basis)
    gen_params[:, 1]  = uniform(bor, w -  bor, num_basis)

    #B_Construct Covariance:
    sigma_unr = np.array( [[sigma_x**2, 0.], [0., sigma_y**2]] )
    R0 = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])#Rotation matrix
    cov = np.dot(R0, np.dot(sigma_unr, np.transpose(R0)))

	#C_We select sigma_x, sigma_y from a random distribution
    gen_params[:, 2:4] = multivariate_normal(center, cov, num_basis)[:, :]
  
	#D_We select wave vector angle (4) and phase inside cosine
    gen_params[:, 4] = uniform(-pi/2., pi/2., num_basis)
    gen_params[:, 5] = uniform(0., pi/2., num_basis)
	
    #Create the gabor set
    for i in xrange(num_basis):
        gabor_set[i, :] = gabor_kern(gen_params[i], gabor_shape).reshape(h*w)     

    return gabor_set, gen_params

###############################################################################


def to_fit(to_estimate, *args):
    """

    """

    gabor_shape, data = args
    return np.sum((gabor_kern(to_estimate, gabor_shape) - data)**2)


def check_bounds(a, b, side):
    """

    """

    if np.sum(np.array([a, b]) <= np.float(side-1)) < 2:
        return False
    elif np.sum(np.array([a, b]) >= 0.) < 2:
        return False
    else:
        return True


def best_approx(a, b, side):
    """

    """

    if a < 0.:
        ret_a = 0
    elif a > (side-1):
        ret_a = side-1
    else:
        ret_a = a
    if b < 0.:
        ret_b = 0
    elif b > (side-1):
        ret_b = side-1
    else:
        ret_b = b
    return [ret_a, ret_b]



def gabor_fit(data, gabor_shape):
    """


    """

    #Assure that both dimensions are equal:
    h, w = gabor_shape
    if h != w:
	    print 'Equal dimensions needed'
	    exit(1)

    #The real borders of the picture are separated 
    #from the canvas by (this is where the real 
    #patch is, the rest is and artifact of the 
    #convolution:

    bor = 2
    margin = h - bor
    ######## Estimate initial cond.##########

    ########Estimator 1############
    #Position of the absolut maximum
    max_pos = np.argmax(np.abs(data))

    x0 = max_pos//h
    x1 = max_pos%h

    ###########Estimator2 ############
    #Wave-vector (Via FFT)
    mod_trans = np.abs(fft2(data))

    #We centered it according to the fftshift function
    mod_trans_s = fftshift(mod_trans)
    max_vec = np.argsort(mod_trans_s.flatten())[-2:]

    delta_ver = -0.5*(max_vec[0]//w -max_vec[1]//w)
    delta_hor = 0.5*(max_vec[0]%w -max_vec[1]%w)

    #Because of the naming conventions we have to do this sort of weird association
    k_x = (2*pi*delta_ver)/w
    k_y = (2*pi*delta_hor)/w


    #Also, because of the way in which we operate on the poles of the Fourie Transform,
    # we need to change the sign of k_x and k_y 
    if k_y < 0:
	    k_x = -k_x
	    k_y = -k_y

    k0 = np.sqrt(k_x**2 + k_y**2)

    if k_y != 0.:
	    psi = np.arctan(k_x/k_y)
    else:
	    psi = pi/2.

    ########### Estimator3 ############
    #Amplitud, phase

    V_1 = data[x0, x1]
    add_x0 = np.int(np.round((pi/2.)*(k_x/(k0**2)))) #(vertical shift)
    add_x1 = np.int(np.round(-(pi/2.)*(k_y/(k0**2)))) #(horizontal shift)

#########EP
    if not check_bounds(x0 + add_x0, x1 + add_x1, h):
        add_x0 = -add_x0
        add_x1 = -add_x1
        if not check_bounds(x0 + add_x0, x1 + add_x1, h):
        #If after the heuristics it's still broken, clamp it.
            a,b = best_approx(x0 + add_x0, x1 + add_x1, h)
            V_2 = data[a,b]
        else:
            V_2 = -data[x0 + add_x0, x1 + add_x1]
    else:
        V_2 = data[x0 + add_x0, x1 + add_x1]


    add2_x0 = -np.int(np.round(2*pi*(k_x/(k0**2))))
    add2_x1 = np.int(np.round(2*pi*(k_y/(k0**2))))

#########EP

    if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
        add2_x0 = -add2_x0
        add2_x1 = -add2_x1
        if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
            #If after the heuristics it's still broken, clamp it.
            a,b = best_approx(x0 + add2_x0, x1 + add2_x1, h)
            V_3 = data[a, b]
        else:
            V_3 = data[x0 + add2_x0, x1 + add2_x1]
    else:
        V_3 = data[x0 + add2_x0, x1 + add2_x1]



    #We need to correct for small deviations in V_3, sign should be that of V1.
    if np.sign(V_3) != np.sign(V_1):
	    V_3 = -V_3
     
    factor = pow((V_3/V_1), 1./16.)
    V_2 = V_2/factor

    #The amplitud is given by:
    A = np.sqrt(V_1**2 + V_2**2)

    #An the angle by: (although it's not giving an accurate estimate)
    #There might be a bug in this estimate.
    if V_1 != 0:
	    tau = np.mod(np.arctan(V_2/V_1) - (k_x*x0 + k_y*x1) , 2*pi)
	    if tau > pi:
		    tau -= pi
    else: 
	    tau = np.mod( pi/2. - (k_x*x0 + k_y*x1), 2*pi)
	    if tau > pi:
		    tau -= pi

    ########### Estimator4 ############
    #Sigma matrix
    V_1 = data[x0, x1]

    add2_x0 = -np.int(np.round(4*pi*(k_x/(k0**2))))
    add2_x1 = np.int(np.round(4*pi*(k_y/(k0**2))))
#########EP

    if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
        add2_x0 = np.round(0.5*add2_x0)
        add2_x1 = np.round(0.5*add2_x1)
        if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
            add2_x0 = -add2_x0
            add2_x1 = -add2_x1
            if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
                #If after the heuristics it's still broken, clamp it.
                a,b = best_approx(x0 + add2_x0, x1 + add2_x1, h)
                V_2 = data[a, b]
            else:
                V_2 = data[x0 + add2_x0, x1 + add2_x1]
        else:
            V_2 = data[x0 + add2_x0, x1 + add2_x1]
    else:
        V_2 = data[x0 + add2_x0, x1 + add2_x1]
                    


    if np.sign(V_2) != np.sign(V_1):
	    V_2 = -V_2

    add3_x0 = np.round(6*(k_y))
    add3_x1 = np.round(6*k_x)
#########EP
    if not check_bounds(x0 + add3_x0, x1 + add3_x1, h):
        #If it's out of range, then it's not fittable
        a,b = best_approx(x0 + add2_x0, x1 + add2_x1, h)
        V_3 = data[a, b]
    else:
        V_3 = data[x0 + add3_x0, x1 + add3_x1]


    if np.sign(V_3) != np.sign(V_1):
	    V_3 = -V_3

    sigma_x = (4*pi/k0)*np.sqrt(1./(-2.*np.log(V_2/V_1)))
    sigma_y = np.sqrt(-18./np.log(V_3/V_1))

    #Put the estimators together in params:
    params = np.empty(8)
    params[0] = x0
    params[1] = x1
    params[2] = sigma_x
    params[3] = sigma_y
    params[4] = psi
    params[5] = tau
    params[6] = k0
    params[7] = A

    ######### Minimizing squared error. Gradient Method.##########
    params2fit = [0, 1, 2, 3, 4, 5, 6, 7]

    num_iter_0 = 3
    num_iter_1 = 3
    num_iter_2 = 20
    threshold = 10.

    valey_array = np.empty([num_iter_0, num_iter_1, num_iter_2, len(params2fit)])
    to_select = np.empty([num_iter_0, num_iter_1, num_iter_2])
    params_ = params.copy()
    i = 0
    j = 0


    #Just to be sure that we're on the right valey, we iterate over the two loops below
    for j in xrange(num_iter_0):
	    params_[0] = params[0] + sigma_x*(-1. + ((2.*j)/(num_iter_0-1)))


	    for k in xrange(num_iter_1):
		    params_[1] = params[1] + sigma_y*(-1 + ((2.*k)/(num_iter_1-1)))

		    for i in xrange(num_iter_2):
			    params_[5] = (2*pi/(num_iter_1-1))*i

			    #here we store the parameters found
			    valey_array[j, k, i, :] = fmin_powell(to_fit, params_, (gabor_shape, data), disp=0)

			    #Here the function's value
			    to_select[j, k, i] = to_fit(valey_array[j, k, i, :], gabor_shape, data)

			    if to_select[j, k, i] < threshold:
				    break

		    if to_select[j, k, i] < threshold:
			    break

	    if to_select[j, k, i] < threshold:
		    break


    if (i == num_iter_2 - 1) and (j == num_iter_0 - 1) and (k == num_iter_1 - 1):
	    min_ind = np.argmin(to_select)

	    j_min = min_ind//(num_iter_1*num_iter_2)
	    r_1 = min_ind%(num_iter_1*num_iter_2)
	    k_min = r_1//num_iter_2
	    i_min = r_1%num_iter_2
    else:
	    j_min = j
	    k_min = k
	    i_min = i

    min_params = valey_array[j_min, k_min, i_min, :] 
    #taking the angles between 0 and 2pi
    min_params[5] = np.mod(min_params[5], 2*pi) 
    return min_params





#For now, it still requires an autotable instance to store the
#results (from proc 0), soon it will not perform input output operations!
#All processes != 0 pass any valid python object instead of to_analyze
#Ugly... to improve soon
def pgabor_fit(RF_data, to_analyze, comm=MPI.COMM_WORLD):
    """

    """

    num_RF, gabor_size, _ = RF_data.shape
    gabor_shape = (gabor_size, gabor_size)

    #We split the workload
    my_lb, my_ub = sca_assim_load(num_RF)
    my_num_RF = my_ub - my_lb

    #We store the value of to_fit function per data point:
    my_error_vector = np.empty(my_num_RF)

    #The number of parameters that we get out is hardcoded and = 8 (gabor params)
    my_obt_params = np.empty([my_num_RF, 8])

    #We defined globally (due to the fair_asim_gather funtion)
    obt_params = np.empty([num_RF, 8])
    error_vector = np.empty(num_RF)

    for i in xrange(my_lb, my_ub):
        data = RF_data[i]
        my_obt_params[i-my_lb, :] = gabor_fit(data, gabor_shape)
        diff = data-gabor_kern(my_obt_params[i-my_lb], gabor_shape)
        my_error_vector[i-my_lb] = np.sum(np.abs(diff))
    comm.Barrier()

    #We gather at root=0
    fair_asim_gather(my_obt_params, num_RF, obt_params)
    fair_asim_gather(my_error_vector, num_RF, error_vector)    

    #Only processor cero performs input output operations
    if comm.rank == 0:

        for i in xrange(obt_params.shape[0]):
            to_analyze.append('params_all', obt_params[i])        
        to_analyze.append('error_vector_all', error_vector)

        #We also correct for artifacts in the fitting process
        obt_par, error_vec = fit_conv(obt_params, error_vector, gabor_size)

        for i in xrange(obt_par.shape[0]):
            to_analyze.append('params_red', obt_par[i])        
        to_analyze.append('error_vector_red', error_vec)
        to_analyze.h5.flush()
    


#All the magnitudes in here ship as phase(0) and radio(1)!
#Allways expects a 2 dim array (num_points, 2)
def phase_analisys(params_array, fig_path, threshold=0., extra_array=None, axis=[0, 1]):
    """

    """

    if threshold == 0.:
        to_plot = params_array[:, :]
    else:
        slicer = params_array[:, 1] < threshold
        to_plot = np.empty([len(params_array[:, 0][slicer]), 2])
        to_plot[:, 0] = params_array[:, 0][slicer]
        to_plot[:, 1] = params_array[:, 1][slicer]

    fig = pyplot.figure(1, figsize=(8, 8))
    #The params below ensure that only the upper part of the circle is shown
    ax = fig.add_axes([0.1, -0.98, 0.8, 2.], polar=True)
    pyplot.scatter(to_plot[:, 0], to_plot[:, 1], c='blue', linewidths=0.0)
    ticks_pos = np.round(np.array(), to_plot[i][2])
    pyplot.yticks(ticks_pos)


    if extra_array != None:
        pyplot.scatter(extra_array[:, 0], extra_array[:, 1], c='black', linewidths=0.0)        
        ax.legend(('recovered distribution', 'ground truth'),
           'upper left', shadow=True)


    #Set axis limits
    x0 = axis[0]
    x1 = axis[1]
    pyplot.ylim(x0, x1)

    fig.savefig(fig_path)
    pyplot.clf()
    pyplot.close(1)


def esp_power_analysis(params_array, fig_path, add_array=None, axis=[0., inf], error_vector=None, add='bare'):
    """

    """
    to_plot = params_array.copy()
    fig = pyplot.figure(1, figsize=(8, 8))
    #The params below ensure that the image fits in the canvas
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    #Below we plot nx = sigma_x*frec, ny = sigma_y*frec

    #TODO: Set fontsize globally 
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize('xx-large')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize('xx-large')

    ax.set_xlabel('nx', size=20)
    ax.set_ylabel('ny', size=20)
    #Assigning the error vector in greyscale to each datapoint
    if error_vector != None:
        pyplot.scatter(np.prod(to_plot[:, [2, 0]], axis=1) , np.prod(to_plot[:, [2, 1]], axis=1), c=error_vector, marker='o', cmap=pyplot.cm.Greys, linewidths=0.0)
    #Or printing only in blue the data in params array and black the extra data given (add_array)
    else:		
      	pyplot.scatter(np.prod(to_plot[:, [2, 0]], axis=1) , np.prod(to_plot[:, [2, 1]], axis=1), c='red', marker='o', linewidths=0.0)
        #Plot another data set on top
        if add_array != None:
            if add=='bare': pyplot.scatter(add_array[:, 0], add_array[:, 1], c='blue', marker=(8, 2 ,0))
            else: pyplot.scatter(np.prod(add_array[:, [2, 0]], axis=1) , np.prod(add_array[:, [2, 1]], axis=1), c='black', marker=(8, 2 ,0))
            ax.legend(('recovered distribution', 'ground truth'), 'upper right', shadow=True)
            

    #Set axis limits
    x0 = axis[0]
    x1 = axis[1]
    if x1 != inf:
        pyplot.xlim(x0, x1)
        pyplot.ylim(x0, x1)

    fig.savefig(fig_path)
    pyplot.close(1)	


def fit_conv(params, error_vector, image_size):
    """

    """
    para = params.copy()
    error_vec = error_vector.copy()
    #Discard the miningless sigmas and wave vectors (hardcoded for now!)
    bound = (image_size)**2
    for i in (2, 3, 6):
        para[:, i] = np.abs(para[:, i])
        slicer = para[:, i] < bound
        para = para[slicer, :]
        error_vec = error_vector[slicer]

    #Shift angles and get frecuency from wave vector
    para[:, 4] = np.mod((para[:, 4] + pi/2.), pi)
    para[:, 5] = np.mod(para[:, 5] , 2*pi)
    para[:, 6] = (para[:, 6])/(2*pi)
    return (para, error_vec)    



def thre_from_error(error_vector, discard_frac):
    """

    """
    num_to_fit = len(error_vector)
    return np.sort(error_vector)[-np.round(num_to_fit*(discard_frac))]





##TODO This properties are to be in an object :params_path: expects full .... .h5 path,
##visual_path: base working directory .../
def off_line_plot(params_path, visual_path, discard_frac, option=2, axis=[0., inf], name=None):
    """

    """

    to_analyze = openFile(params_path, 'r')
    error_v = to_analyze.root.error_vector_red[:][0]
    params_a = to_analyze.root.params_red[:, :]
    #Plot
    off_plot(params_a, error_v, discard_frac, option, axis=axis, name=name, save_dir=visual_path)
    to_analyze.close()


def off_plot(params, error, discard_frac, option, axis=[0., inf], name=None, save_dir=None, params2=None, add='bare'):
    """

    """

    #setting threshold
    actual_threshold = thre_from_error(error, discard_frac)
    params_array = params[(error <= actual_threshold), :]
    error_vector = error[(error <= actual_threshold)]

    if name == None: fig_name = ctime() + '_' + '.png'
    else: fig_name = name 

    if save_dir == None: visual_path = getcwd()+'/'
    else: visual_path = save_dir

    extra_opt = None #The default is None
    extra_wave = None # ""
    
    if params2 != None:
        if add == 'bare': #To use with direct experimental data
            extra_opt = params2[:, :]
        else: 
            extra_opt = params2[:, [2, 3, 6]]
            extra_wave = params2[:, [4, 6]]             

    if option==0 or option==2: #A_wave vector
        fig_path = visual_path + 'wave_vector_' + fig_name
        phase_analisys(params_array[:, [4, 6]], fig_path, extra_array=extra_wave)

    if option==1 or option==2: #B_optical power
        fig_path = visual_path + 'optical_power_' + fig_name
        esp_power_analysis(params_array[:, [2, 3, 6]], fig_path, add_array=extra_opt, axis=axis, add=add)

    if option==3:#C_optical power + colorscale of errors
        fig_path = visual_path + 'optical_power_grey_scale_' + fig_name
        esp_power_analysis(params_array[:, [2, 3, 6]], fig_path, add_array=extra_opt, axis=axis, error_vector=-error_vector, add=add)
    
###############################################################################

def fft_set(input_path, save_path):
    """

    """
    h5 = openFile(input_path, 'r')
    ##Check in here whether patches is a member, also check for number of elements
    fullimset = h5.root.patches
    num, a, b = fullimset.shape
    fullimset = fullimset[:, : ,:]
    h5.close()

    S = np.empty([a//2, b])

    for i in xrange(num):
        A = fft2(fullimset[i,0:(a//2),:]-fullimset[i,(a//2):a,:])
        ####################
        if np.sum(A) != np.sum(A):
            exit("NAN encountered! Closing")
        ####################
        S += np.abs(fftshift(fft2(fullimset[i,0:(a//2),:]-fullimset[i,(a//2):a,:])))
        printG(i)

    S = np.abs(S)/num

    savefig(save_path)




#this function generates all the active units (out of H') up to gamma (including it)
def compute_active_units(gamma, H_prime):
    """

    """
    full_list = []
    for i in xrange(gamma+1):
	    full_list.append([])
    #We find the subgrups of i different numbers taken out of H_prime possible
    #For i going from 1 ... to H_prime. 
    for i in xrange(gamma+1):
	    for s in combinations(xrange(H_prime), i):
		    full_list[i].append(np.array(s, dtype=np.int8))
    return full_list




def pseudo_joint(y, W_ro, pi_equivalent, sigma, gamma_, ro):
    """

    """

    Max = np.power(np.sum(W_ro, axis=1), (1./ro))
    Max_y = Max - y
    noise_contr =-0.5*sigma*np.inner(Max_y, Max_y)
    prior_contr = gamma_*pi_equivalent
    return noise_contr + prior_contr



def Hartley_kernel(k_x, k_y, s, s_out):
    """

    """
    gx, gy = np.ogrid[0:s_out, 0:s_out]
    K = (2*np.pi*(k_x*gx + k_y*gy))/s
    K = np.sin(K) + np.cos(K)
    return K + np.abs(np.min(K)) 

    #Up to what spacial frecuency.
    k_max = np.int(np.sqrt(num_patches))



def compute_average():
    pass

def compute_variance():
    pass

