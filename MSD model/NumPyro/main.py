# Import required packages
import sys
import os
import math
import time
import copy
import random as pyrandom
import logging
import pickle as pickle
import pprint as pp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import arviz as az
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
from scipy.stats import rankdata
import torch
import numpyro
from numpyro.infer import HMC, MCMC, NUTS, DiscreteHMCGibbs, MixedHMC, HMCGibbs
import numpyro.distributions as dist
from jax import ops, lax, random, jit, nn
from jax.scipy.special import gammaln
import jax.numpy as jnp
from utils import *

# Set plotting style
plt.style.use('seaborn-darkgrid')
sns.set()  # nice plot aesthetic


class TreeDist(dist.Distribution):
    def __init__(self, tree_class, index, tau, mu=None, sigma=None):
        self.index = index
        self.tau = tau
        self.leaf_nodes = jnp.array(tree_class.leaf_nodes)
        self.num_nodes = tree_class.num_nodes
        self.optype = tree_class.optype
        if(self.optype == 'class'): 
            self.y_classes = jnp.array(tree_class.y_classes)
        else:
            self.mu = mu
            self.sigma = sigma
        super(TreeDist, self).__init__()

    def traverse(self, x, index, split): 
        node_id = 0
        node_id = lax.while_loop(lambda node_id: jnp.isin(node_id,self.leaf_nodes,invert=True), lambda node_id: self.get_children_id(node_id, x, split, index), node_id)
        return (node_id).astype(int)

    def get_children_id(self,node_id,x,split,index): #### TODO: make sure node_id and split align
        tmp = 2 * (node_id + 1)
        return (jnp.where(x[index[node_id]] <= split[node_id], tmp - 1, tmp)).astype(int)

    def body_fun(self,i,leaf_values,data):
        leaf_id = self.traverse(data["x_train"][i],self.index,self.tau)
        leaf_values = ops.index_update(leaf_values,ops.index[leaf_id,i],data["y_train"][i])
        return leaf_values

    def log_prob(self, data): # TODO make more efficient memory wise for leaf_values
        if(self.optype == 'class'): # classification dataset
            leaf_values = jnp.full([self.num_nodes,len(data['x_train'])],-1) # jnp.nan
            leaf_values = lax.fori_loop(0,data['n_train'],lambda i,val: self.body_fun(i,val,data),leaf_values)
            logllh_total = 0 # variable to store total log-likelihood value over all leaves in tree
            for i in range(self.num_nodes): 
                freqs = hist_count(leaf_values[i,:],self.y_classes)
                N = jnp.sum(freqs)
                leaf_llh = dist.DirichletMultinomial(np.array([5/2]*len(self.y_classes)),total_count=N) ### TODO make generic alpha input
                logllh_total += leaf_llh.log_prob(freqs) - (gammaln(N + 1) - (gammaln(freqs + 1)).sum(axis=-1, keepdims=True))
        else: # regression dataset NOTE: uses likelihood as described in (Denison, et al. 1998)
            leaf_values = jnp.full([self.num_nodes,len(data['x_train'])],jnp.nan)
            leaf_values = lax.fori_loop(0,data['n_train'],lambda i,val: self.body_fun(i,val,data),leaf_values)
            logllh_total = 0 # variable to store total log-likelihood value over all leaves in tree
            leaf_indx = 0 # keep track of leaves TODO: make sure this aligns for non-symmetric trees
            for i in range(self.num_nodes): 
                if (i in self.leaf_nodes):
                    leaf_llh = dist.Normal(self.mu[leaf_indx],self.sigma[leaf_indx])
                    logllh_total += jnp.nansum(leaf_llh.log_prob(leaf_values[i,:]))
                    leaf_indx += 1
        return  logllh_total 

# Define generic Bayesian tree class
class BayesianTree(object):
    def __init__(self,data,settings):
        # Create information for tree structure
        # TODO: check this parameterisation is consistent with non symmetric trees
        self.treeGraph = None
        self.optype = settings.optype
        self.num_nodes = len(settings.tree_struct)
        self.leaf_nodes = [i for i, e in enumerate(settings.tree_struct) if e == -1] # based on integer value
        self.internal_nodes = [i for i, e in enumerate(settings.tree_struct) if e == 0]
        self.nodes = sorted(self.internal_nodes + self.leaf_nodes)
        self.node_info = {}
        # self.depth = ?? # TODO
        self.nx = data['nx'] # dimension of predictors x
        self.sr = self.calculate_spearman_rank(data)
        if(self.optype == 'class'):
            self.y_classes = np.unique(data['y_train'])
        else:
            pass

    def calculate_spearman_rank(self,data):
        data_ranked = rankdata(data['x_train'], axis=0)
        sr = jnp.abs(jnp.corrcoef(data_ranked,rowvar=False)) # only absolute value is required for probability calculations
        sr = ops.index_update(sr,ops.index[jnp.less(sr,0.3)],0)# remove 'spurious small correlations' (as per (Pratola, 2016))
        return sr

    def model_tree(self,data):
        if(self.optype == 'class'): # classification dataset
            index = numpyro.sample('index', dist.Categorical(probs=np.tile(np.array([1/data['nx']]*data['nx']),(len(self.internal_nodes),1))))
            tau = numpyro.sample('tau', dist.Beta([1]*len(self.internal_nodes),[1]*len(self.internal_nodes)))
            return numpyro.sample('p', TreeDist(self, index, tau), obs=data)
        if(self.optype == 'real'): # regression TODO make hyperparameters inputs!!!!!!! Currently defaults from (Chipman , et al. 1998)
            index = numpyro.sample('index', dist.Categorical(probs=np.tile(np.array([1/data['nx']]*data['nx']),(len(self.internal_nodes),1))))
            tau = numpyro.sample('tau', dist.Beta(jnp.ones(len(self.internal_nodes)),jnp.ones(len(self.internal_nodes))))
            sigma = numpyro.sample('sigma', dist.InverseGamma(np.tile(3/2,(len(self.leaf_nodes))),np.tile(3/2,(len(self.leaf_nodes)))))
            mu = numpyro.sample('mu', dist.Normal(jnp.zeros(len(self.leaf_nodes)),sigma))
            return numpyro.sample('p', TreeDist(self, index, tau, mu, sigma), obs=data)
        else:
            print("Optype must be either class or real (for classification/regression respectively). Exiting program.")
            exit()

    # TODO: ASK ADRIAN - CAN THE GIBBS UPDATE ONLY DEPEND ON OTHER VARIABLES AND NOT ON DATA?!
    def gibbs_fn(self, rng_key, gibbs_sites, hmc_sites):
        tau = hmc_sites['tau']
        mu = hmc_sites['mu']
        sigma = hmc_sites['sigma']
        index = gibbs_sites['index']

        # Select node to change variable 
        node = random.choice(rng_key,jnp.array(self.internal_nodes))
        # node = np.random.choice(self.internal_nodes)

        # Define indicator function for whether valid split exists ## TODO

        # Calculate probability of changing index - based on Spearman correlation matrix - as defined in (Pratola 2016) Eq 7.
        probs_change = self.sr[index[node],:]/jnp.sum(self.sr[index[node],:])
        new_index = random.choice(rng_key,jnp.array(self.internal_nodes),p=probs_change)
        index = ops.index_update(index,ops.index[node],new_index)

        # TODO MH ACCEPTANCE STEP NEEDS TO BE INCLUDED HERE

        # How to best update the index given tau, mu, sigma
        return {'index': index}
        
    def get_children_id(self,node_id):
        tmp = 2 * (node_id + 1)
        return (tmp - 1, tmp) 
            
    def plot_traces(self, samples):
        # Plots tau traces conditional on index 
        if(self.nx > 1): # check if more than one predictor variable
            for indx in np.unique(samples["index"]):
                for var in [var for var in samples.keys() if var != "index"]:
                    az.plot_trace({str(var)+' | i = '+str(indx):samples[var]})
                    plt.show()
        else:
            for var in samples.keys():
                az.plot_trace({str(var):samples[var]})
                plt.show()

    def plotLikelihood(self,data,title=""):
        ## TODO: make generic
        # Plots likelihood function with respect to continuous splitting variable
        # Visualise likelihood functions
        N = 101
        x = data['x_train']
        y = data['y_train']
        llh = np.zeros(N)
        mu_vals = jnp.column_stack( ( 5*jnp.ones(N), jnp.linspace(0,10,N) ) )# TODO: make generic
        # mu_vals = jnp.column_stack( ( jnp.linspace(0,10,N), 5*jnp.ones(N) ) ) 

        tree = TreeDist(self,index=jnp.array([0]),tau=jnp.array([0.2]))
        for i in range(len(mu_vals)):
            tree.mu = mu_vals[i,:]
            llh[i] = tree.log_prob(data)

        plt.figure()
        plt.plot(mu_vals[:,1],llh)
        plt.show()

    def graph_tree(self):
        G = nx.DiGraph()

        # Add edges when current node has child nodes
        for node in self.internal_nodes:
            for child in self.get_children_id(node):
                G.add_edge(str(node),str(child))
        
        self.treeGraph = G

    def draw_tree(self,tree_vars=None,ax=None):
        plot_now = False
        if (self.treeGraph is None):
            self.graph_tree()

        if(ax is None):
            plot_now = True
            plt.figure()
            ax = plt.gca()
        pos = graphviz_layout(self.treeGraph, prog="dot")
        if(tree_vars is None):
            nx.draw(self.treeGraph, pos=pos, ax=ax, with_labels=True, node_size=300)
        else:
            label_dict = {}
            leaf_indx = 0 # counter for labeling terminal leaves
            node_indx = 0 # counter for labeling internal nodes
            for node in self.nodes:
                if(node in self.internal_nodes):
                    label_dict[str(node)] = "x"+str(tree_vars['index'][0,node_indx])+" < "+"{:.2f}".format(tree_vars['tau'][node_indx])
                    node_indx += 1
                else:
                    if(self.optype == 'real'):
                        label_dict[str(node)] = "mu = "+"{:.2f}".format(tree_vars['mu'][leaf_indx])
                        leaf_indx += 1
                    else:
                        label_dict[str(node)] = "L"+str(leaf_indx)
                        leaf_indx += 1

            nx.draw(self.treeGraph, pos=pos, ax=ax, labels=label_dict, with_labels=True, node_size=500)
        if(plot_now):
            plt.show()
        
    def report_traces(self, mcmc, theta_true, MCMC_est=False, kde=False, forest=False, autocorr=False):
        trace = az.from_numpyro(mcmc)
        # Plot trace for each variable
        self.plot_traces(trace.posterior)

        if(MCMC_est):
            # Plot the estimate for the mean of log(τ) cumulating mean
            logtau = np.log(trace.posterior["tau"])
            mlogtau = [np.mean(logtau.values[0][:i]) for i in np.arange(1, logtau.size)]
            plt.figure(figsize=(15, 4))
            plt.axhline(np.log(theta_true['tau']), lw=2.5, color="gray")
            plt.plot(mlogtau, lw=2.5)
            plt.ylim(np.min(logtau), np.max(logtau))
            plt.xlabel("Iteration")
            plt.ylabel("MCMC mean of log(tau)")
            plt.title("MCMC estimation of log(tau)")
            plt.show()
            
        if(kde):
            az.plot_posterior(trace)
            plt.show()
            
        if(forest):
            az.plot_forest(trace,r_hat=True)
            plt.show()
            
        if(autocorr):
            az.plot_autocorr(trace)
            plt.show()

def main():
    time_0 = time.process_time()
    settings = process_command_line()
    print('%'*160)
    print('Beginning main.py')
    print('Current settings:')
    pp.pprint(vars(settings))

    # Initialise numpy random seed
    np.random.seed(settings.init_id * 1000)

    # Loading data
    print('\nLoading data ...')
    data = load_data(settings)
    data = process_dataset(data)
    print('Loating data ... completed')
    print('Dataset name = %s' % settings.dataset)
    print('Characteristics of the dataset:')
    print('n_train = %d, n_test = %d, n_dim = %d' %\
            (data['n_train'], data['n_test'], data['nx']))
    if settings.optype == 'class':
        print('n_class = %d' % (data['n_class']))

    # plot_dataset(data)
    
    # Initialise probabilistic model  
    print('\nInitialising HMC\n')

    # Define tree object
    tree = BayesianTree(data=data,settings=settings)

    tree.draw_tree()
    # tree.gibbs_fn(random.PRNGKey(0),{'index':[2,1,0]},[])
    
    kernel = DiscreteHMCGibbs(NUTS(tree.model_tree, forward_mode_differentiation=True))#,random_walk=True) # if random_walk=True, applies MH update instead of Gibbs
    # hmc_kernel = NUTS(tree.model_tree, forward_mode_differentiation=True)
    # kernel = HMCGibbs(hmc_kernel, gibbs_fn=tree.gibbs_fn, gibbs_sites=['index'])
    # kernel = MixedHMC(HMC(tree.model_tree, trajectory_length=0.1, forward_mode_differentiation=True), num_discrete_updates=20)
    # TODO write custom gibbs update function --> include rotations
    mcmc= MCMC(kernel, num_samples=1000, num_warmup=1000, num_chains=4)
    mcmc.run(random.PRNGKey(0),data)
    samples = mcmc.get_samples()
    mcmc.print_summary()

    summary_plot(mcmc,data,tree)   

    # tree.report_traces(mcmc,data['theta_true'],MCMC_est=True)

    pass


if __name__ == "__main__":
    main()
