#==============================================================================
# Nathan Vaughn
# Schroedinger Equation Integral Form Numerical Solver
# METHOD: Inverse Iteration for Morse Potential
# 06/13/16
#==============================================================================
#%%  STARTUP
import numpy as np
import scipy as sp
from scipy.sparse import linalg
from scipy.interpolate import interp1d
np.set_printoptions(precision=16)
import matplotlib.pyplot as plt
import time
import os

start_time = time.time()
#%% Initialize
D           = 4
#omega       = 0.204124
omega       = 1/np.sqrt(2*D)
nmax        = 4

a           = 5
xmax        = 4*a       # maximum radius, the "effective infinity"
xmin        = -a
nx_sweep    = 200        # number of grid points
nx          = nx_sweep
dx          = (xmax-xmin)/nx      # grid spacing

#dr          = 0.02
#nr          = 300
#rmax        = int(2*nr/3)*dr
#rmin        = -int(nr/3)*dr

#%% Functions
def E_true(n):
    return -D + (n+1/2) - 1/(4*D)*(n+1/2)**2
    
def x_vec():
    x_vec = np.linspace(xmin+dx/2,xmax-dx/2,nx)
    return x_vec

def x(nx):
    return xmin + (nx+1/2)*dx

def V(x):
    return D*(np.exp(-2*omega*x) -2*np.exp(-omega*x))
    
def G(x,E):
    kappa = np.sqrt(2*(-E))
    k = 1j*kappa
    return np.real(np.exp(1j*k*x)*(-1j/(2*k)))  
    
def matrix(E):
    A = np.fromfunction(lambda i, ii: V(x(ii))*G(np.abs(x(i)-x(ii)),E), (nx,nx))
    A = 2*A*dx
    return A

def inverse_iteration(A):
    I = np.eye(len(A))
    vec_old = np.random.rand(len(A))
    vec_old = vec_old/np.linalg.norm(vec_old)
    err = 1
    eig_old = 2
    LU = sp.linalg.lu_factor(A-I)
    count=0
    while err > 1e-15: # 1e-14 sufficeint for Morse, 1e-16 doesnt improve
        vec_new     = sp.linalg.lu_solve(LU,vec_old)        
        vec_new     = vec_new/np.linalg.norm(vec_new)
        eig_new     = np.dot(vec_new,np.dot(A,vec_new))
        err         = np.abs(eig_new-eig_old)
        eig_old     = eig_new
        vec_old     = vec_new
        count+=1
#        if count%1000 == 0:
#            print('Inv_It count = %g' %count)
#            print('eig = %g' %eig_new)
#            print('error = %g' %err)
        if count > 3000 and np.abs(eig_new-1)>0.03:
            break  
        if count > 5000: # SUfficient for Morse.  20000 doesn't improve
            break
    return vec_new, eig_new  
       
    

def Energy_Sweep(Emin,Emax,nE):
#    Evec = np.linspace(Emin,Emax,nE)
    Esplit1 = (Emin+Emax)/2
    Esplit2 = (8*Emax + 2*Emin)/10
    Esplit3 = (19*Emax + Emin)/20
    Evec = np.linspace(Emin,Esplit1,int(0.25*nE))
    Evec = np.append(Evec,np.linspace(Esplit1,Esplit2,int(nE*0.25)))
    Evec = np.append(Evec,np.linspace(Esplit2,Esplit3,int(nE*0.25)))
    Evec = np.append(Evec,np.linspace(Esplit3,Emax,int(nE*0.25)))
    eig_vals = np.zeros(len(Evec))
    c = 0
    for E in Evec:
#        print('Energy = %g ' %E)
        A = matrix(E)
        wave,eigenvalue = inverse_iteration(A)
        eig_vals[c] = eigenvalue
#        print('eig closest to unity = %e' %eigenvalue)
        c += 1
    plt.figure()
    plt.plot(Evec,eig_vals,'k.',label='Sweep Values')
#    plt.plot(E_true(0),1,'rx',mew=3,label='Analytic Values')
    count=0
    for n in range(2*D):
            if E_true(n) < Emax and E_true(n) > Emin:
                if count==0:
                    plt.plot(E_true(n),1,'rx',mew=3,label='Analytic Values')
                else:
                    plt.plot(E_true(n),1,'rx',mew=3)
                count += 1
##    plt.legend(loc=2)
#    plt.title('Morse Potential Energy Sweep')
    plt.title('Energy Sweep: $x_{max} = %g$, $x_{min} = %g$' %(xmax,xmin))
    plt.xlabel('Energy')
    plt.ylabel('Eigenvalue Closest to One')
    plt.xlim([-4, 0.5])
#    plt.close()
#    plt.legend(loc='upper left')
    
    return Evec, eig_vals


def Energy_Log_Sweep(expmin,expmax,nE):
    Evec = -np.logspace(expmin,expmax,nE,endpoint=True, base=10)
    eig_vals = np.zeros(len(Evec))
    c = 0
    for E in Evec:
        print('Energy = %g' %E)
        A = matrix(E)
        wave, eigenvalue = inverse_iteration(A)
        eig_vals[c] = eigenvalue
        print('eig closest to unity = %e' %eigenvalue)
        c += 1
#    plt.figure()
#    plt.xlim([10**-3, 13])
#    plt.semilogx(np.abs(Evec),eig_vals,'r.')
#    Emax = -10**expmax
#    Emin = -10**expmin
#    for n in range(24):
#        if E_true(n) < Emax and E_true(n) > Emin:
#            plt.semilogx(np.abs(E_true(n)),1,'bx',mew=3,label='n=%g' %n)

#    plt.legend(loc=1)
#    plt.title('Morse Potentail Logarithmic Energy Sweep')
#    plt.xlabel('log(|E|)')
#    plt.ylabel('Eigenvalue Closest to One')
    
    return Evec, eig_vals

def E_crossings(eig_vals,Evec,gap):
    E_cross = []
    E_jumps = []
#    E_low_vec = []
#    E_high_vec = []
    for i in range(len(eig_vals)-1):
        if (eig_vals[i]-1)*(eig_vals[i+1]-1) < 0:
            if np.abs(eig_vals[i] - eig_vals[i+1]) < gap:
                E_cross.append((Evec[i]+Evec[i+1])/2)
            else:
                E_jumps.append((Evec[i]+Evec[i+1])/2)
    E_cross = np.array(E_cross)
    E_jumps = np.array(E_jumps)
    return E_cross, E_jumps
            


def Energy_Bisect(E_low,E_high):
    flag = 0
    # left
    A = matrix(E_low)
#    eig_left = eig_closest_to_one(A)
    wave_left,eig_left = inverse_iteration(A)
    # right
    A = matrix(E_high)
#    eig_right = eig_closest_to_one(A)
    wave_right,eig_right = inverse_iteration(A)
    
    # mid
    E_mid = (E_high+E_low)/2
    A = matrix(E_mid)
    wave_mid,eig_mid = inverse_iteration(A)
    
    if (eig_right-1)*(eig_left-1) > 0:
        print('eig_left and _eig_right didn\'t cross 1.')
        flag = 1
        return
 
      
    # Approximate Energy with a linear fit    
    slope = (eig_right-eig_left)/(E_high-E_low)
    E_guess = E_low + (1-eig_left)/(slope)


    if (eig_left-1)*(eig_mid-1) < 0:
        E_high = E_mid
    else:
        E_low = E_mid
                

    return E_low, E_high, E_guess, flag 
    
def Energy_guess(E_low,E_high,nx):
    nx = int(nx)
    E_old   = 0
    niter   = 30
    count   = 0
    flag    = 0
    print('Exact Energy = %2.16g' %(E_exact))
    print('nx = %g' %nx)
    print('dx = %g' %dx)
    print('xmax = %g' %xmax)
    print('xmin = %g' %xmin)
    print()
    while count < niter:
        temp_time = time.time()
        E_low,E_high,E_guess,flag = Energy_Bisect(E_low,E_high)
#        print('Count = %g' %count)
#        print('Energy Approximation = %2.14g' %(E_guess/C))
        if np.abs((E_old-E_guess)) < 1e-15:
            break
        E_old = np.copy(E_guess)
#        if flag == 1:
#            break
        count +=1
    # refine the new E_low and E_high for the next grid size
    print('Energy Guess   = %2.16g' %E_guess)
    print('Absolute Error = %2.14g' %(np.abs(E_guess-E_exact)))
    print('Relative Error = %2.14g' %(np.abs((E_guess-E_exact)/E_exact)))
    print("--- %s seconds ---" % (time.time() - temp_time))
    print()
    print()
    if flag ==1:
        return
    A =  matrix(E_guess)
    wave_guess,eig_guess = inverse_iteration(A)
    return E_guess, wave_guess
        
def RE(E_coarse,E_fine,order,k):
    # order is the previous order of convergence
    # k is the step size reduction factor.  2 if dx was halved.
    return (k**order*E_fine-E_coarse)/(k**order-1)    


#%% Run for Energy Sweep

#a           = 5
#rmax        = 4*a       # maximum radius, the "effective infinity"
#rmin        = -a
#nr          = 400        # number of grid points
#dr          = (rmax-rmin)/nr      # grid spacing
#
#E_start = -D
#E_end   = -0.0001
##target = 2
##E_start = 1.05*E_true(target)
##E_end = 0.95*E_true(target)
#nE      = 400
#dE      = (E_end-E_start)/nE
###Evec, eig_vals = Energy_Sweep(-12,-0.001,500)
#Evec, eig_vals = Energy_Sweep(E_start,E_end,nE)
##Evec, eig_vals = Energy_Sweep(-6.6,-3,200)

#%%
#a           = 5
#rmax        = 4*a       # maximum radius, the "effective infinity"
#rmin        = -2*a
#nr          = 800        # number of grid points
#dr          = (rmax-rmin)/nr      # grid spacing
#
#E_start = -3.0
#E_end   = -0.36
#nE      = 200
##dE      = (E_end-E_start)/nE
##Evec, eig_vals = Energy_Sweep(-12,-0.001,500)
#Evec, eig_vals = Energy_Sweep(E_start,E_end,nE)
##Evec, eig_vals = Energy_Log_Sweep(np.log10(-E_start),np.log10(-E_end),300)
#
##%%
#a           = 5
#rmax        = 8*a       # maximum radius, the "effective infinity"
#rmin        = -4*a
#nr          = 400        # number of grid points
#dr          = (rmax-rmin)/nr      # grid spacing
#
#E_start = -0.3
##E_start = -0.03
#E_end   = -0.0001
#nE      = 20
#dE      = (E_end-E_start)/nE
##Evec, eig_vals = Energy_Sweep(-12,-0.001,500)
#Evec, eig_vals = Energy_Sweep(E_start,E_end,nE)
##Evec, eig_vals = Energy_Log_Sweep(np.log10(-E_start),np.log10(-E_end),nE)


#%%
#E_cross, E_jumps = E_crossings(eig_vals,Evec,0.05)
#E_jumps = np.append(E_start,E_jumps)
#E_jumps = np.append(E_jumps,E_end)
#os.system('say "Ping"')


#%% 3 stages of R.E.


#rmax        = 55       # maximum radius, the "effective infinity"
#rmin        = -5
#nr          = 400        # number of grid points
#dr          = (rmax-rmin)/nr      # grid spacing
#
#target = 6
##E_start = 1.05*E_true(target)
#E_end = 0.93*E_true(target)
#E_start = 1.1*E_true(target)
##E_end = 0.95*E_true(target)
#nE      = 5
#dE      = (E_end-E_start)/nE
#Evec, eig_vals = Energy_Sweep(E_start,E_end,nE)
##plt.close()
#
#E_cross, E_jumps = E_crossings(eig_vals,Evec,0.05)
#E_jumps = np.append(E_start,E_jumps)
#E_jumps = np.append(E_jumps,E_end)
#
#
#nr_coarse = nr
#k = 2
#n=target # lowest energy hits
#E_final = []
#absolute_errs = []
#relative_errs = []
#nr_orig = np.copy(nr)
#for i in range(len(E_cross)):
#    E = E_cross[i]
#    E_low = E_cross[i]-1*dE
#    E_high = E_cross[i]+1*dE
#    E_exact = E_true(n)   
#    
#
#    E_vec = []
#    E_vec_2 = []
#    E_vec_3 = []
#    
#    # Stage 1
#    nr = nr_coarse
#    dr      = (rmax-rmin)/(nr)
#    E_temp = Energy_guess(E_low,E_high,nr)
#    E_vec = np.append(E_vec,E_temp)
#    err_abs = np.abs(E_temp-E_exact)
#    E_low   = E_exact-err_abs
#    E_high  = E_exact+err_abs
#    
#    # Stage 2
#    nr = k*nr_coarse
#    dr      = (rmax-rmin)/(nr)
#    E_temp = Energy_guess(E_low,E_high,nr)
#    E_vec = np.append(E_vec,E_temp)
#    err_abs = np.abs(E_temp-E_exact)
#    E_low   = E_exact-err_abs
#    E_high  = E_exact+err_abs
#    E_vec_2 = np.append(E_vec_2,RE(E_vec[0],E_vec[1],2,k))
#    
##    E_final_temp = RE(E_vec[0],E_vec[1],2,k)
#    
#    # Stage 3
#    nr = k**2*nr_coarse
#    dr      = (rmax-rmin)/(nr)
#    E_temp = Energy_guess(E_low,E_high,nr)
#    E_vec = np.append(E_vec,E_temp)
#    err_abs = np.abs(E_temp-E_exact)
##    E_low   = E_exact-err_abs
##    E_high  = E_exact+err_abs
#    E_vec_2 = np.append(E_vec_2,RE(E_vec[1],E_vec[2],2,k))
#    E_vec_3 = np.append(E_vec_3,RE(E_vec_2[0],E_vec_2[1],4,k))
#    
##    E_final_temp = RE(E_vec_2[0],E_vec_2[1],4,k)
##    
##     #Stage 4
#    nr = k**3*nr_coarse
#    dr      = (rmax-rmin)/(nr)
#    E_temp = Energy_guess(E_low,E_high,nr)
#    E_vec = np.append(E_vec,E_temp)
#    E_vec_2 = np.append(E_vec_2,RE(E_vec[2],E_vec[3],2,k))
#    E_vec_3 = np.append(E_vec_3,RE(E_vec_2[1],E_vec_2[2],4,k))
#    
#    E_final_temp = RE(E_vec_3[0],E_vec_3[1],6,k)
##
##
#    E_final.append(E_final_temp)
#    absolute_errs.append(np.abs((E_final_temp-E_exact)))
#    relative_errs.append(np.abs((E_final_temp-E_exact)/E_exact))
#    
#    n+=1
##    
#E_final = np.array(E_final)
#absolute_errs = np.array(absolute_errs)
#relative_errs = np.array(relative_errs)
##computed_n_squared = 1/np.sqrt(-E_final/C)
##
#print('Results for (nr,rmax,rmin) = (%g, %g, %g)' %(nr_orig,rmax,rmin))
#print()
#print('Final Energies')
#print(E_final)
##print()
##print('Computed n values')
##print(computed_n_squared)
##print()
#print('Absolute Errors for Each Energy:')
#print(absolute_errs)
#print('Relative Errors for Each Energy:')
#print(relative_errs)
#
#
#print("--- %s seconds ---" % (time.time() - start_time))
#os.system('say "Ping"')
#





#%% FIX dr, use VARIABLE rmax And nr
## Do a sweep for each nr.  Slower than ideal, but should help matters
#start_time = time.time()
#
#
#a           = 5
##rmax        = 45       # maximum radius, the "effective infinity"
#xmin        = -5
##nr          = 800        # number of grid points
##dr          = (rmax-rmin)/nr      # grid spacing
##dr          = 0.06875
##dr          = 0.05
##nr          = int(round((rmax-rmin)/dr))
#
#target = 1
#E_exact = E_true(target)
#
#
##dr          = 0.05
##nr = int(1.25*round((6.2*np.exp(0.0928*target)-rmin)/dr))
##rmax = rmin + nr*dr
#
#
#xmax = 7.75*np.exp(0.1*target)
##xmax = 1.25*6.2*np.exp(0.0928*target)
#nx=200
#dx = (xmax-xmin)/nx
#
##E_start = E_true(1.04*target)
##E_end   = E_true(0.96*target)   
##E_start = 1.08*E_true(target)
##E_end = 0.85*E_true(target)
##E_start = E_exact - 0.2
##E_end = E_exact+0.2
#
### For low E
#E_low = E_exact - 0.22
#E_high = E_exact+0.2
#
### for high E
##E_low = E_exact - 0.022
##E_high = E_exact+0.01
##E_low = 1.2*E_exact
##E_high = 0.8*E_exact
#
##nE      = 3
##dE      = (E_end-E_start)/nE
#
#nx_coarse = nx
#k = 2
#n=target # lowest energy hits
#E_final = []
#absolute_errs = []
#relative_errs = []
#nr_orig = np.copy(nx)
#
#    
#
#E_vec = []
#E_vec_2 = []
#E_vec_3 = []
#    
#    # Stage 1
#nx      = nx_coarse
#dx      = (xmax-xmin)/(nx)
##Evec, eig_vals = Energy_Sweep(E_start,E_end,nE)
##E_cross, E_jumps = E_crossings(eig_vals,Evec,0.05)
##E = E_cross
##E_low = E_cross-1*dE
##E_high = E_cross+1*dE
#
#E_temp = Energy_guess(E_low,E_high,nx)
#E_vec = np.append(E_vec,E_temp)
#err_abs = np.abs(E_temp-E_exact)
#
#
#    # Stage 2
#nx = k*nx_coarse
#dx      = (xmax-xmin)/(nx)
##Evec, eig_vals = Energy_Sweep(E_start,E_end,nE)
##E_cross, E_jumps = E_crossings(eig_vals,Evec,0.05)
##E = E_cross
##E_low = E_cross-1*dE
##E_high = E_cross+1*dE
#
#E_temp = Energy_guess(E_low,E_high,nx)
#E_vec = np.append(E_vec,E_temp)
#err_abs = np.abs(E_temp-E_exact)
#E_vec_2 = np.append(E_vec_2,RE(E_vec[0],E_vec[1],2,k))
#    
##E_final_temp = RE(E_vec[0],E_vec[1],2,k)
#    
#    # Stage 3
#nx = k**2*nx_coarse
#dx      = (xmax-xmin)/(nx)
##Evec, eig_vals = Energy_Sweep(E_start,E_end,nE)
##E_cross, E_jumps = E_crossings(eig_vals,Evec,0.05)
##E = E_cross
##E_low = E_cross-1*dE
##E_high = E_cross+1*dE
#
#E_temp = Energy_guess(E_low,E_high,nx)
#E_vec = np.append(E_vec,E_temp)
#err_abs = np.abs(E_temp-E_exact)
#E_vec_2 = np.append(E_vec_2,RE(E_vec[1],E_vec[2],2,k))
#E_vec_3 = np.append(E_vec_3,RE(E_vec_2[0],E_vec_2[1],4,k))
#
##E_final_temp = RE(E_vec_2[0],E_vec_2[1],4,k)
#   
#    # Stage 4
#nx = k**3*nx_coarse
#dx      = (xmax-xmin)/(nx)
##Evec, eig_vals = Energy_Sweep(E_start,E_end,nE)
##E_cross, E_jumps = E_crossings(eig_vals,Evec,0.05)
##E = E_cross
##E_low = E_cross-1*dE
##E_high = E_cross+1*dE
#
#E_temp = Energy_guess(E_low,E_high,nx)
#E_vec = np.append(E_vec,E_temp)
#E_vec_2 = np.append(E_vec_2,RE(E_vec[2],E_vec[3],2,k))
#E_vec_3 = np.append(E_vec_3,RE(E_vec_2[1],E_vec_2[2],4,k))
#
#E_final_temp = RE(E_vec_3[0],E_vec_3[1],6,k)
#
#
#E_final.append(E_final_temp)
#absolute_errs.append(np.abs((E_final_temp-E_exact)))
#relative_errs.append(np.abs((E_final_temp-E_exact)/E_exact))
#
##    
#E_final = np.array(E_final)
#absolute_errs = np.array(absolute_errs)
#relative_errs = np.array(relative_errs)
##computed_n_squared = 1/np.sqrt(-E_final/C)
##
#print('Results for (nx,xmax,xmin) = (%g, %g, %g)' %(nr_orig,xmax,xmin))
#print()
#print('Final Energies')
#print(E_final)
##print()
##print('Computed n values')
##print(computed_n_squared)
##print()
#print('Absolute Errors for Each Energy:')
#print(absolute_errs)
#print('Relative Errors for Each Energy:')
#print(relative_errs)
#
#
#print("--- %s seconds ---" % (time.time() - start_time))
#os.system('say "Ping"')


#%%  Wavefunction Plotter
#
#rmax = 55
#rmin = -5
#nr = 200
#dr = (rmax-rmin)/nr
#
#E0 = E_true(0)
#E7 = E_true(7)
#
#A0 = matrix(E0)
#A7 = matrix(E7)
#
#wave0,eig0 = inverse_iteration(A0)
#wave7,eig7 = inverse_iteration(A7)
#
#rvec = r_vec()
#
#plt.figure()
#plt.title('Potential Well and Energy Levels')
#plt.plot(rvec[8:],V(rvec[8:]),'k',label='Potential Well')
#plt.plot(rvec[14:21],E0*np.ones(len(rvec[14:21])),'r',label='Ground State E')
#plt.plot(rvec[10:63],E7*np.ones(len(rvec[10:63])),label='Excited State E')
#plt.xlim([rmin,rmax])
#plt.ylabel('Wavefunction')
#plt.legend()
#plt.figure()
#plt.subplot(2,1,1)
#plt.title('Wavefunctions')
#plt.plot(rvec,wave0,label='Ground State')
#plt.xlim([rmin,rmax])
#plt.ylabel('Wavefunction')
#plt.legend()
#plt.subplot(2,1,2)
#plt.plot(rvec,wave7,label='7th Excited State')
#plt.ylabel('Wavefunction')
#plt.xlim([rmin,rmax])
#plt.xlabel('Domain')
#plt.legend()



#%%  Plot wavefunction tails
#shift=15
#shift2 = 168
#plt.figure()
#plt.subplot(2,1,1)
#plt.title('Comparison of the Decaying Tails')
#plt.plot(rvec[-shift2:],wave0[-shift2:],label='Ground State')
#plt.xlim([rvec[-shift2],rvec[-1]])
#plt.ylabel('Wavefunction')
#plt.legend()
#plt.subplot(2,1,2)
#plt.plot(rvec[-shift:],wave7[-shift:],label='7th Excited State')
#plt.ylabel('Wavefunction')
#plt.xlim([rvec[-shift],rvec[-1]])
#plt.xlabel('Domain')
#plt.legend(loc='best')




#%% Simply using the extrapolation from finite square...
target = 0
E_exact = E_true(target)
#E_low = 1.3*E_exact
#E_high = 0.9*E_exact

E_low = -4
E_high = -3.5

xmax = 15
xmin = -5


nx_coarse = 150
grf = 3 # grid refinement factor
E_vec=[]
E_vec_2 = []
E_vec_3 = []
wave_errs = []
wave_errs_2 = []
wave_errs_3 = []

nx=nx_coarse
dx = (xmax-xmin)/(nx)
xvec = x_vec()
#wave_true = wave_analytic(n)


# Stage 1
nx = nx_coarse
dx = (xmax-xmin)/(nx)
x1 = x_vec()
# gets rid of the uncertain final digit, which comes in to play when it's multiplied thousands of times
E_temp, wave1 = Energy_guess(E_low,E_high,nx)
#if np.linalg.norm(np.abs(wave1+wave_true)) < np.linalg.norm(np.abs(wave1-wave_true)):
#    wave1 = -wave1
E_vec = np.append(E_vec,E_temp)
err_abs = np.abs(E_temp-E_exact)
#E_low   = E_exact-1.0*err_abs
#E_high  = E_exact+1.0*err_abs



# Stage 2
nx = grf*nx_coarse
dx = (xmax-xmin)/(nx)
x2 = x_vec()
input_wave = np.interp(x2,x1,wave1)
E_temp, wave2 = Energy_guess(E_low,E_high,nx)
#if np.linalg.norm(np.abs(wave2[1::grf]+wave_true)) < np.linalg.norm(np.abs(wave2[1::grf]-wave_true)):
#    wave2 = -wave2
E_vec = np.append(E_vec,E_temp)
err_abs = np.abs(E_temp-E_exact)
E_low   = E_exact-1.0*err_abs
E_high  = E_exact+1.0*err_abs
#E_low   = E_exact-0.5*err_abs
#E_high  = E_exact+0.5*err_abs

# Stage 3
nx = grf**2*nx_coarse
dx = (xmax-xmin)/(nx)
x3 = x_vec()
input_wave = np.interp(x3,x2,wave2)
E_temp, wave3 = Energy_guess(E_low,E_high,nx)
#if np.linalg.norm(np.abs(wave3[1+grf::grf**2]+wave_true)) < np.linalg.norm(np.abs(wave3[1+grf::grf**2]-wave_true)):
#    wave3 = -wave3
E_vec = np.append(E_vec,E_temp)
err_abs = np.abs(E_temp-E_exact)
E_low   = E_exact-1.0*err_abs
E_high  = E_exact+1.0*err_abs
#E_low   = E_exact-0.5*err_abs
#E_high  = E_exact+0.5*err_abs

# Stage 4
nx = grf**3*nx_coarse
dx = (xmax-xmin)/(nx)
x4 = x_vec()
input_wave = np.interp(x4,x3,wave3)
E_temp, wave4 = Energy_guess(E_low,E_high,nx)
#if np.linalg.norm(np.abs(wave4[1+grf+grf**2::grf**3]+wave_true)) < np.linalg.norm(np.abs(wave4[1+grf+grf**2::grf**3]-wave_true)):
#    wave4 = -wave4
E_vec = np.append(E_vec,E_temp)


# Perform Extrapolations
ecr = 2 # energy convergence rate
E_vec_2 = np.append(E_vec_2,RE(E_vec[0],E_vec[1],ecr,grf))
E_vec_2 = np.append(E_vec_2,RE(E_vec[1],E_vec[2],ecr,grf))
E_vec_2 = np.append(E_vec_2,RE(E_vec[2],E_vec[3],ecr,grf))

E_vec_3 = np.append(E_vec_3,RE(E_vec_2[0],E_vec_2[1],2*ecr,grf))
E_vec_3 = np.append(E_vec_3,RE(E_vec_2[1],E_vec_2[2],2*ecr,grf))

E_final = RE(E_vec_3[0],E_vec_3[1],ecr*3,grf)

#wcr = 2 # wave converegence rate
#wave_extrapolated1 = RE(wave1,wave2[1::grf],wcr,grf)
#wave_extrapolated2 = RE(wave2[1::grf],wave3[1+grf::grf**2],wcr,grf)
#wave_extrapolated3 = RE(wave3[1+grf::grf**2],wave4[1+grf+grf**2::grf**3],wcr,grf)
#wave_extrapolated4 = RE(wave_extrapolated1,wave_extrapolated2,wcr*2,grf)
#wave_extrapolated5 = RE(wave_extrapolated2,wave_extrapolated3,wcr*2,grf)
#wave_final = RE(wave_extrapolated4,wave_extrapolated5,wcr*3,grf)


## Error results
#
#print('ENERGY RESULTS')
#print('--------------')
#print('init abserr = %2.6g' %(np.abs(E_vec[0]-E_exact)))
#print('ref1 abserr = %2.6g' %(np.abs(E_vec[1]-E_exact)))
#print('ref2 abserr = %2.6g' %(np.abs(E_vec[2]-E_exact)))
#print('ref3 abserr = %2.6g' %(np.abs(E_vec[3]-E_exact)))
#print()
#print('Ext1 abserr = %2.6g' %(np.abs(E_vec_2[0]-E_exact)))
#print('Ext2 abserr = %2.6g' %(np.abs(E_vec_2[1]-E_exact)))
#print('Ext3 abserr = %2.6g' %(np.abs(E_vec_2[2]-E_exact)))
#print()
#print('Ext4 abserr = %2.6g' %(np.abs(E_vec_3[0]-E_exact)))
#print('Ext5 abserr = %2.6g' %(np.abs(E_vec_3[1]-E_exact)))
#print()
#print('Final error = %2.6g' %(np.abs(E_final-E_exact)))
#print()
#print()
#%% Error results

print('ENERGY RESULTS')
print('--------------')
print('nx: %g  error = %2.6g' %(nx_coarse,np.abs(E_vec[0]-E_exact)))
print('nx: %g  error = %2.6g, ratio =%g' %(grf*nx_coarse,(np.abs(E_vec[1]-E_exact)), np.abs(E_vec[0]-E_exact)/np.abs(E_vec[1]-E_exact)))
print('nx: %g error = %2.6g, ratio =%g' %(grf**2*nx_coarse,(np.abs(E_vec[2]-E_exact)), np.abs(E_vec[1]-E_exact)/np.abs(E_vec[2]-E_exact)))
print('nx: %g error = %2.6g, ratio =%g' %(grf**3*nx_coarse,(np.abs(E_vec[3]-E_exact)), np.abs(E_vec[2]-E_exact)/np.abs(E_vec[3]-E_exact)))
print()
print('1st Extrap error = %2.6g' %(np.abs(E_vec_2[0]-E_exact)))
print('1st Extrap error = %2.6g, ratio =%g' %((np.abs(E_vec_2[1]-E_exact)), np.abs(E_vec_2[0]-E_exact)/np.abs(E_vec_2[1]-E_exact)))
print('1st Extrap error = %2.6g, ratio =%g' %((np.abs(E_vec_2[2]-E_exact)), np.abs(E_vec_2[1]-E_exact)/np.abs(E_vec_2[2]-E_exact)))
print()
print('2nd Extrap error = %2.6g' %(np.abs(E_vec_3[0]-E_exact)))
print('2nd Extrap error = %2.6g, ratio =%g' %((np.abs(E_vec_3[1]-E_exact)), np.abs(E_vec_3[0]-E_exact)/np.abs(E_vec_3[1]-E_exact)))
print()
print('Final error = %2.6g' %(np.abs(E_final-E_exact)))
print()
print()

