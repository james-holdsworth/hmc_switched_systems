"""
Simulate a first order state space model given by
x_{t+1} = a*x_t + b*u_t + w_t
y_t = x_t + e_t
with q and r the standard deviations of w_t and q_t respectively

Jointly estimate the smoothed state trajectories and parameters theta = {a, b, q, r}
to give p(x_{1:T}, theta | y_{1:T})

GOAL:
Use a Monte Carlo style approach to perform MPC where the state constraints are satisfied
with a given probability.

Current set up: Uses MC to give an expected cost and then satisfies,
Chance state constraints using a log barrier formulation
Chance state constraints using a log barrier formulation
Input constraints using a log barrier formulation

Implementation:
Uses custom newton method to solve
Uses JAX to compile and run code on GPU/CPU and provide gradients and hessians
"""

## TODO list
# refactor the parameters a little bit to put them into a dictionary, so that the same function can be used for the ssm simulator and the jax mpc simulator


# general imports
from numpy.core.numeric import zeros_like
import pystan
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_trace, col_vec, row_vec
from pathlib import Path
import pickle

# jax related imports
import jax.numpy as jnp
from jax import grad, jit, device_put, jacfwd, jacrev
from jax.ops import index, index_add, index_update
from jax.config import config

# optimisation module imports (needs to be done before the jax confix update)
# from optimisation import log_barrier_cost, solve_chance_logbarrier

config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy


#----------------- Parameters ---------------------------------------------------#

Ns = 200             # number of samples we will use for MC MPC

# simulation parameters
T = 100             # number of time steps to simulate and record measurements for
z1_0 = 3.0            # initial x
z2_0 = 0.0
r1_true = 0.1        # measurement noise standard deviation
r2_true = 0.01
q1_true = 0.05       # process noise standard deviation
q2_true = 0.005      # process noise standard deviation
m_true = 2
b_true = 0.7
k_true = 0.25

Nx = 2;
Ny = 2;
Nu = 1;

#----------------- Simulate the system-------------------------------------------#

# def ssm1(x1, x2, u, T, a11=0.0, a12=1.0, b1=0.0):
#     return (a11*x1 + a12*x2 + b1*u)*T
# def ssm2(x1, x2, u, T, a21=0.5, a22=-0.5, b2=0.3):
#     return a21*x1 + a22*x2 + b2*u
def ssm_euler(x,u,A,B,T):
    return (np.matmul(A,x) + np.matmul(B,u)) * T;

# SSM equations
A1 = np.zeros((Nx,Nx), dtype=float)
B1 = np.zeros((Nx,Nu), dtype=float)

A2 = np.zeros((Nx,Nx), dtype=float)
B2 = np.zeros((Nx,Nu), dtype=float)

A1[0,1] = 1.0;
A1[1,0] = -k_true/m_true
A1[1,1] = -b_true/m_true
B1[1,0] = 1/m_true;

# mass quadruples!
A2[0,1] = 1.0;
A2[1,0] = -k_true/m_true/4
A2[1,1] = -b_true/m_true/4
B2[1,0] = 1/m_true/4;



z_sim = np.zeros((Nx,T+1), dtype=float) # state history

# initial state
z_sim[0,0] = z1_0 
z_sim[1,0] = z2_0 

# noise predrawn and independant
w_sim = np.zeros((Nx,T),dtype=float)
w_sim[0,:] = np.random.normal(0.0, q1_true, T)
w_sim[1,:] = np.random.normal(0.0, q2_true, T)

# create some inputs that are random but held for 10 time steps
u = np.random.uniform(-0.5,0.5, T*Nu)
u = np.reshape(u, (Nu,T))

# spicy
t_fail = 50
for k in range(T):
    # x1[k+1] = ssm1(x1[k],x2[k],u[k]) + w1[k]
    # x2[k+1] = ssm2(x1[k],x2[k],u[k]) + w2[k]
    if k<t_fail:
        z_sim[:,k+1] = z_sim[:,k] + ssm_euler(z_sim[:,k],u[:,k],A1,B1,1.0) + w_sim[:,k]
    else:
        z_sim[:,k+1] = z_sim[:,k] + ssm_euler(z_sim[:,k],u[:,k],A2,B2,1.0) + w_sim[:,k]

# simulate measurements
v = np.zeros((Ny,T), dtype=float)
v[0,:] = np.random.normal(0.0, r1_true, T)
v[1,:] = np.random.normal(0.0, r2_true, T)
y = np.zeros((Ny,T), dtype=float)
y[0,:] = z_sim[0,:-1]
y[1,:] = (-k_true*z_sim[0,:-1] -b_true*z_sim[1,:-1] + u[0,:])/m_true
y = y + v; # add noise

plt.subplot(2,1,1)
plt.plot(u[0,:])
plt.plot(y[1,:],linestyle='None',color='r',marker='*')
plt.title('Simulated inputs and measurement used for inference')
plt.subplot(2, 1, 2)
plt.plot(z_sim[0,:])
plt.plot(y[0,:],linestyle='None',color='r',marker='*')
plt.title('Simulated state 1 and measurements used for inferences')
plt.tight_layout()
plt.show()

#----------- USE HMC TO PERFORM INFERENCE ---------------------------#
# avoid recompiling
model_name = 'LSSM_O2_MSD_demo'
path = 'stan/'
if Path(path+model_name+'.pkl').is_file():
    model = pickle.load(open(path+model_name+'.pkl', 'rb'))
else:
    model = pystan.StanModel(file=path+model_name+'.stan')
    with open(path+model_name+'.pkl', 'wb') as file:
        pickle.dump(model, file)

stan_data = {
    'N':T,
    'y':y,
    'u':u,
    'O':Nx,
    'D':Ny,
    'T':1.0
}

fit = model.sampling(data=stan_data, warmup=1000, iter=2000)
traces = fit.extract()

# state samples
z_samps = np.transpose(traces['z'],(1,0,2)) # Ns, Nx, T --> Nx, Ns, T


# parameter samples
m1_samps = traces['m1'].squeeze()
k1_samps = traces['k1'].squeeze()
b1_samps = traces['b1'].squeeze() # single valued parameters shall 1D numpy objects! The squeeze has been squoze
q1_samps = np.transpose(traces['q1'],(1,0)) 
r1_samps = np.transpose(traces['r1'],(1,0))
t_samps = traces['t'].squeeze()
m2_samps = traces['m2'].squeeze()
k2_samps = traces['k2'].squeeze()
b2_samps = traces['b2'].squeeze() # single valued parameters shall 1D numpy objects! The squeeze has been squoze
q2_samps = np.transpose(traces['q2'],(1,0)) 
r2_samps = np.transpose(traces['r2'],(1,0))

# plot the initial parameter marginal estimates
q1plt1 = q1_samps[0,:].squeeze()
q2plt1 = q1_samps[1,:].squeeze()
r1plt1 = r1_samps[0,:].squeeze()
r2plt1 = r1_samps[1,:].squeeze()

# plot the initial parameter marginal estimates
q1plt2 = q2_samps[0,:].squeeze()
q2plt2 = q2_samps[1,:].squeeze()
r1plt2 = r2_samps[0,:].squeeze()
r2plt2 = r2_samps[1,:].squeeze()


plot_trace(m1_samps,2,4,1,'m')
plt.title('HMC inferred parameters')
plot_trace(k1_samps,2,4,2,'k')
plot_trace(b1_samps,2,4,3,'b')
plot_trace(q1plt1,2,4,4,'q1')
plot_trace(q2plt1,2,4,5,'q2')
plot_trace(r1plt1,2,4,6,'r1')
plot_trace(t_samps,2,4,7,'t')
plt.show()

plot_trace(m2_samps,2,4,1,'m')
plt.title('HMC inferred parameters')
plot_trace(k2_samps,2,4,2,'k')
plot_trace(b2_samps,2,4,3,'b')
plot_trace(q1plt2,2,4,4,'q1')
plot_trace(q2plt2,2,4,5,'q2')
plot_trace(r1plt2,2,4,6,'r1')
plot_trace(r2plt2,2,4,7,'r2')
plt.show()

# plot some of the initial marginal state estimates
for i in range(4):
    if i==1:
        plt.title('HMC inferred position')
    plt.subplot(2,2,i+1)
    plt.hist(z_samps[0,:,i*20+1],bins=30, label='p(x_'+str(i+1)+'|y_{1:T})', density=True)
    plt.axvline(z_sim[0,i*20+1], label='True', linestyle='--',color='k',linewidth=2)
    plt.xlabel('x_'+str(i+1))
plt.tight_layout()
plt.legend()
plt.show()






