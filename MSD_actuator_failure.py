"""
Simulates a mass-spring-damper system in one dimension (2 state system) with random noise on the state transition. 
Generates measurements of the system corresponding to position and acceleration of the mass, with random noise on each measurement.
At time t, the system model changes from system1 --> system 2, in the simplest case this is a change in the parameters.
Given a window of data where the switch occurs at time t, pystan will sample the states, parameters for both systems, and switching time t jointly.
"""

# general imports
from numpy.core.numeric import zeros_like
import pystan
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_trace
from pathlib import Path
import pickle

plot_bool = False
#----------------- Parameters ---------------------------------------------------#

T = 100             # number of time steps to simulate and record measurements for

# true (simulation) parameters
z1_0 = 3.0  # initial position
z2_0 = 0.0  # initial velocity
r1_true = 0.1 # measurement noise standard deviation
r2_true = 0.01
q1_true = 0.05 # process noise standard deviation
q2_true = 0.005 
m_true = 2
b_true = 0.7
k_true = 0.25

Nx = 2; # number of states
Ny = 2; # number of measurements
Nu = 2; # number of inputs

#----------------- Simulate the system-------------------------------------------#

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
B1[1,1] = 1/m_true;

# actuators have an issue
s1 = 0.25;
s2 = 0;
A2[0,1] = 1.0;
A2[1,0] = -k_true/m_true
A2[1,1] = -b_true/m_true
B2[1,0] = s1/m_true # s1 and s2 modify the actuator gain
B2[1,1] = s2/m_true

z_sim = np.zeros((Nx,T+1), dtype=float) # state history allocation

# load initial state
z_sim[0,0] = z1_0 
z_sim[1,0] = z2_0 

# noise is predrawn and independant
w_sim = np.zeros((Nx,T),dtype=float)
w_sim[0,:] = np.random.normal(0.0, q1_true, T)
w_sim[1,:] = np.random.normal(0.0, q2_true, T)

# create some inputs that are random
u = np.random.uniform(-0.5,0.5, T*Nu)
u = np.reshape(u, (Nu,T))

# time of switch
t_switch = 100 # T = t_switch implies no failure
for k in range(T):
    # x1[k+1] = ssm1(x1[k],x2[k],u[k]) + w1[k]
    # x2[k+1] = ssm2(x1[k],x2[k],u[k]) + w2[k]
    if k<t_switch:
        z_sim[:,k+1] = z_sim[:,k] + ssm_euler(z_sim[:,k],u[:,k],A1,B1,1.0) + w_sim[:,k]
    else:
        z_sim[:,k+1] = z_sim[:,k] + ssm_euler(z_sim[:,k],u[:,k],A2,B2,1.0) + w_sim[:,k]

# draw measurement noise
v = np.zeros((Ny,T), dtype=float)
v[0,:] = np.random.normal(0.0, r1_true, T)
v[1,:] = np.random.normal(0.0, r2_true, T)

# simulated measurements 
y = np.zeros((Ny,T), dtype=float)
y[0,:] = z_sim[0,:-1]
y[1,:t_switch] = (-k_true*z_sim[0,:t_switch] -b_true*z_sim[1,:t_switch] + u[0,:t_switch] + u[1,:t_switch])/m_true
y[1,t_switch:] = (-k_true*z_sim[0,t_switch:-1] -b_true*z_sim[1,t_switch:-1] + s1*u[0,t_switch:] + s2*u[1,t_switch:])/(m_true) # s1, s2 modified inputs
y = y + v; # add noise to measurements

if plot_bool:
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
model_name = 'MSD_actuator_failure'
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
m_samps = traces['m'].squeeze()
k_samps = traces['k'].squeeze()
b_samps = traces['b'].squeeze() # single valued parameters shall 1D numpy objects! The squeeze has been squoze
q_samps = np.transpose(traces['q'],(1,0)) 
r_samps = np.transpose(traces['r'],(1,0))
t_samps = traces['t'].squeeze()
s1_samps = traces['s1'].squeeze()
s2_samps = traces['s2'].squeeze()

# plot the initial parameter marginal estimates
q1plt = q_samps[0,:].squeeze()
q2plt = q_samps[1,:].squeeze()
r1plt = r_samps[0,:].squeeze()
r2plt = r_samps[1,:].squeeze()


plot_trace(m_samps,2,5,1,'m')
plt.title('HMC inferred parameters')
plot_trace(k_samps,2,5,2,'k')
plot_trace(b_samps,2,5,3,'b')
plot_trace(q1plt,2,5,4,'q1')
plot_trace(q2plt,2,5,5,'q2')
plot_trace(r1plt,2,5,6,'r1')
plot_trace(r2plt,2,5,7,'r2')
plot_trace(t_samps,2,5,8,'t')
plot_trace(s1_samps,2,5,9,'s1')
plot_trace(s2_samps,2,5,10,'s2')
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






