"""
Simulates a mass-spring-damper system in one dimension (2 state system) with random noise on the state transition. 
Generates measurements of the system corresponding to position and acceleration of the mass, with random noise on each measurement.
At time t, the system model changes from system1 --> system 2, in the simplest case this is a change in the parameters.
Given a window of data where the switch occurs at time t, pystan will sample the states, parameters for both systems, and switching time t jointly.
"""
import os
import platform
# if platform.system()=='Darwin':
#     import multiprocessing
#     multiprocessing.set_start_method("fork")
# general imports
import numpy as np
from numpy.core.numeric import zeros_like
import numpyro
numpyro.set_host_device_count(1)
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from jax import lax, random, jit
from jax.ops import index, index_update, index_add
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
# from helpers import plot_trace
from pathlib import Path
import pickle



# if plot_bool:
#     plt.subplot(2,1,1)
#     plt.plot(u[0,:])
#     plt.title('Simulated inputs and measurement used for inference')
#     plt.subplot(2, 1, 2)
#     plt.plot(z_sim[0,:])
#     plt.plot(y[0,:],linestyle='None',color='r',marker='*')
#     plt.title('Simulated state 1 and measurements used for inferences')
#     plt.tight_layout()
#     plt.show()

# #----------- USE HMC TO PERFORM INFERENCE ---------------------------#
# # avoid recompiling
# data = {
#     'N':T,
#     'y':y,
#     'u':u,
#     'O':Nx,
#     'D':Ny,
#     'T':tau
# }


# class CustomDist(dist.Distribution):
#     def __init__(self):
#         self.r = r 
#         # ETCCCC 
#         pass

#     def log_prob(self,data):
#         r = self.r
#         pass

# before = floor_search(t,1,N); // should return an integer, stan doesn't allow real -> int conversion
    # delt = (t - floor(t))*T; // the timestep within the update 
#         # // state likelihood (apparently much better to do univariate sampling twice)
#     # z[1,2:before] ~ normal(z[1,1:before-1] + T*z[2,1:before-1], q[1]);
#     # z[2,2:before] ~ normal(z[2,1:before-1] + -(k*T/m)*z[1,1:before-1] + -(b*T/m)*z[2,1:before-1] + (T/m)*(u[1,1:before-1]+u[2,1:before-1]), q[2]); // input affects second state only
#     # z_inter[1] = z[1,before] + delt*z[2,before];
#     # z_inter[2] = z[2,before] + -(k*delt/m)*z[1,before] + -(b*delt/m)*z[2,before] + (delt/m)*(u[1,before]+u[2,before]);
#     # z[1,before+1] ~ normal(z_inter[1] + (T - delt)*z_inter[2], (delt/T)*q[1] + (T-delt)*q[1]/T);
#     # z[2,before+1] ~ normal(z_inter[2] + -(k*(T-delt)/m)*z_inter[1] + -(b*(T - delt)/m)*z_inter[2] + ((T - delt)/m)*(s1*u[1,before] + s2*u[2,before]), (delt/T)*q[2] + (T-delt)*q[2]/T); // input affects second state only
#     # z[1,before+2:N] ~ normal(z[1,before+1:N-1] + T*z[2,before+1:N-1], q[1]);
#     # z[2,before+2:N] ~ normal(z[2,before+1:N-1] + -(k*T/m)*z[1,before+1:N-1] + -(b*T/m)*z[2,before+1:N-1] + (T/m)*(s1*u[1,before+1:N-1] + s2*u[2,before+1:N-1]), q[2]); // input affects second state only

# #         // measurement likelihood
# #     y[1,1:before] ~ normal(z[1,1:before], r[1]); // measurement of first state only
# #     // y[2,1:before] ~ normal(-(k/m)*z[1,1:before] - (b/m)*z[2,1:before] + (u[1,1:before]+u[2,1:before])/m, r[2]); // acceleration measurement?
# #     y[1,before+1:N] ~ normal(z[1,before+1:N], r[1]); // measurement of first state only
# #     // y[2,before+1:N] ~ normal(-(k/m)*z[1,before+1:N] - (b/m)*z[2,before+1:N] + (s1*u[1,before+1:N] + s2*u[2,before+1:N])/m, r[2]); // acceleration measurement?
# # }
#         return log_total


def model(data):
    # prior on x
    mu0_1 = data['x0'][0]
    mu0_2 = data['x0'][1]
    sig0_1 = data['sig0'][0]
    sig0_2 = data['sig0'][1]

    # datas
    u = data['u'] # inputs
    y_obs = data['y'] # measurements
    T = data['tau'] # simulation timestep
    Nx = data['x0'].shape[0]
    Nu = u.shape[0]
    Ny = y_obs.shape[0]
    N = y_obs.shape[1] # length of y dictates the length of the data

    # noise std dev priors
    r = numpyro.sample('r',dist.Cauchy(0.0,1.0))
    q_1 = numpyro.sample('q_1',dist.Cauchy(0.0,1.0))
    q_2 = numpyro.sample('q_2',dist.Cauchy(0.0,1.0))
    
    # prior on system parameters
    m = numpyro.sample('m',dist.Normal(2.0,0.1))
    k = numpyro.sample('k',dist.Normal(0.25,0.1))
    b = numpyro.sample('b',dist.Normal(0.7,0.1))

    # prior on failure parameters
    # t = numpyro.sample('t',dist.Uniform(1.0,N))
    # s_1 = numpyro.sample('s1',dist.Uniform()) # in (0,1) is implied
    # s_2 = numpyro.sample('s2',dist.Uniform()) # in (0,1) is implied
    # SSM equations
    param = {
        'T':T,
        'r':r,
        'q_1':q_1,
        'q_2':q_2,
        'm':m,
        'k':k,
        'b':b
        # 't':t,
        # 's_1':s_1,
        # 's_2':s_2
    }

    z = jnp.zeros((Nx,N+1),dtype=float)
    y = jnp.zeros((Ny,N),dtype=float)
    # initial state prior
    z = index_update(z,index[0,0],numpyro.sample('z0_1',dist.Normal(mu0_1,sig0_1)))
    z = index_update(z,index[1,0],numpyro.sample('z0_2',dist.Normal(mu0_2,sig0_2)))
    carry = {
        'z':z,
        'u':u,
        'y_obs':y_obs,
        'y':y,
        'param':param
    }
    iis = jnp.arange(N)
    carry,_ = lax.scan(__model_scanner,carry,iis)
    z = carry['z']
    y = carry['y']
    # return
    # carry,_ = lax.scan(__measurement_scanner,carry,iis)

def __model_scanner(carry,ii):
    param = carry['param']
    mu = __euler_msd(carry['z'][:,ii],carry['u'][:,ii],param)
    carry['z'] = index_update(carry['z'], index[0,ii+1], numpyro.sample('z', dist.Normal(0,param['q_1'])))
    carry['z'] = index_update(carry['z'], index[1,ii+1], numpyro.sample('z', dist.Normal(0,param['q_2'])))
    carry['y'] = index_update(carry['y'],index[0,ii], numpyro.sample('y', dist.Normal(carry['z'][0,ii],carry['param']['r']),obs=carry['y_obs'][0,ii]))
    return carry,[]

def __euler_msd(x,u,param):
    mu = jnp.zeros((x.shape[0]),dtype=float)
    mu = index_update(mu,index[0],x[0] + param['T']*x[1])
    mu = index_update(mu,index[1],x[1] + param['T'] * (-(param['k']/param['m'])*x[0] -(param['b']/param['m'])*x[1] + u[0]/param['m']))
    # z[2,1:before-1] + -(k*T/m)*z[1,1:before-1] + -(b*T/m)*z[2,1:before-1] + (T/m)*(u[1,1:before-1]+u[2,1:before-1])
    return mu


if __name__ == '__main__':
    plot_bool = True
    #----------------- Parameters ---------------------------------------------------#

    T = 50             # number of time steps to simulate and record measurements for
    tau = 0.1
    # true (simulation) parameters
    z1_0 = 3.0  # initial position
    z2_0 = 0.0  # initial velocity
    x0 = np.array([3.0,0.0])
    r1_true = 0.01 # measurement noise standard deviation
    # r2_true = 0.5
    q1_true = 0.1*tau # process noise standard deviation
    q2_true = 0.05*tau 
    sig0 = np.array([q1_true,q2_true])
    m_true = 2
    b_true = 0.7
    k_true = 0.25

    Nx = 2; # number of states
    Ny = 1; # number of measurements
    Nu = 1; # number of inputs

    #----------------- Simulate the system-------------------------------------------#

    def ssm_euler(x,u,A,B,T):
        reps = 1
        for ii in range(reps):
            x = x + (np.matmul(A,x) + np.matmul(B,u)) * T/reps;
        return x

    # SSM equations
    A1 = np.zeros((Nx,Nx), dtype=float)
    B1 = np.zeros((Nx,Nu), dtype=float)

    A2 = np.zeros((Nx,Nx), dtype=float)
    B2 = np.zeros((Nx,Nu), dtype=float)

    A1[0,1] = 1.0;
    A1[1,0] = -k_true/m_true
    A1[1,1] = -b_true/m_true
    B1[1,0] = 1/m_true;
    # B1[1,1] = 1/m_true;

    # actuators have an issue
    # s1 = 0.33;
    # s2 = 0.75;
    # A2[0,1] = 1.0;
    # A2[1,0] = -k_true/m_true
    # A2[1,1] = -b_true/m_true
    # B2[1,0] = s1/m_true # s1 and s2 modify the actuator gain
    # B2[1,1] = s2/m_true

    z_sim = np.zeros((Nx,T+1), dtype=float) # state history allocation

    # load initial state
    z_sim[0,0] = z1_0 
    z_sim[1,0] = z2_0 

    # noise is predrawn and independant
    w_sim = np.zeros((Nx,T),dtype=float)
    w_sim[0,:] = np.random.normal(0.0, q1_true, T)
    w_sim[1,:] = np.random.normal(0.0, q2_true, T)

    # create some inputs that are random
    u = np.random.uniform(-5,5, T*Nu)
    u = np.reshape(u, (Nu,T)) * 0

    # time of switch
    t_switch = 100  # T = t_switch implies no failure
    for k in range(T):
        # x1[k+1] = ssm1(x1[k],x2[k],u[k]) + w1[k]
        # x2[k+1] = ssm2(x1[k],x2[k],u[k]) + w2[k]
        # if k<t_switch:
            z_sim[:,k+1] = ssm_euler(z_sim[:,k],u[:,k],A1,B1,tau) + w_sim[:,k]
        # else:
        #     z_sim[:,k+1] = ssm_euler(z_sim[:,k],u[:,k],A2,B2,tau) + w_sim[:,k]

    # draw measurement noise
    v = np.zeros((Ny,T), dtype=float)
    v[0,:] = np.random.normal(0.0, r1_true, T)
    # v[1,:] = np.random.normal(0.0, r2_true, T)

    # simulated measurements 
    y = np.zeros((Ny,T), dtype=float)
    y[0,:] = z_sim[0,:-1]
    # y[1,:t_switch] = (-k_true*z_sim[0,:t_switch] -b_true*z_sim[1,:t_switch] + u[0,:t_switch] + u[1,:t_switch])/m_true
    # y[1,t_switch:] = (-k_true*z_sim[0,t_switch:-1] -b_true*z_sim[1,t_switch:-1] + s1*u[0,t_switch:] + s2*u[1,t_switch:])/(m_true) # s1, s2 modified inputs
    y = y + v; # add noise to measurements




    data = {
        'x0':x0,
        'sig0':sig0,
        'tau':tau,
        'u':u,
        'y':y
    }
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=1000, num_warmup=1000, num_chains=4)
    mcmc.run(random.PRNGKey(0),data)
    samples = mcmc.get_samples()
    mcmc.print_summary()





# # plot some of the initial marginal state estimates
# for i in range(4):
#     if i==1:
#         plt.title('HMC inferred position')
#     plt.subplot(2,2,i+1)
#     plt.hist(z_samps[0,:,i*20+1],bins=30, label='p(x_'+str(i+1)+'|y_{1:T})', density=True)
#     plt.axvline(z_sim[0,i*20+1], label='True', linestyle='--',color='k',linewidth=2)
#     plt.xlabel('x_'+str(i+1))
# plt.tight_layout()
# plt.legend()
# plt.show()






