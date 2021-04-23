import sys
import math
import numpy as np
import jax.numpy as jnp
from jax import ops
import argparse
from jax.interpreters.xla import DeviceArray # for data conversion if dataset isn't correct type
import matplotlib.pyplot as plt
from scipy.stats import mode


################## Command-line related function ##################
def parser_add_common_options(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', default='toy-class',
            help='name of the dataset  [default: %(default)s]')
    parser.add_argument('--optype', dest='optype', default='class',
            help='nature of outputs in your dataset (class/real) '\
            'for (classification/regression)  [default: %(default)s]')
    parser.add_argument('--data_path', dest='data_path', default='../../datasets/',
            help='path of the dataset')
    parser.add_argument('--debug', dest='debug', default='0', type=int,
            help='debug or not? (1=True/0=False)')
    parser.add_argument('--op_dir', dest='op_dir', default='.', 
            help='output directory for pickle files (NOTE: make sure directory exists) [default: %(default)s]')
    parser.add_argument('--tag', dest='tag', default='', 
            help='additional tag to identify results from a particular run')
    parser.add_argument('--save', dest='save', default=0, type=int,
            help='do you wish to save the results? (1=True/ 0=False)') 
    parser.add_argument('--verbose',dest='verbose', default=1, type=int,
            help='verbosity level (0 is minimum, 4 is maximum)')
    parser.add_argument('--init_id', dest='init_id', default=1, type=int,
            help='init_id (changes random seed for multiple initializations)')
    
    # group = parser.add_argument_group("Prior specification / Hyperparameters")
    # group.add_argument('--prior', dest='prior', default='cgm',
    #         help='nature of prior (cgm for classification, cgm/bart for regression)')
    # group.add_argument('--tree_prior', dest='tree_prior', default='cgm',
    #         help='tree prior that specifies probability of splitting a node'\
    #         ' (only cgm prior has been implemented till now) [default: %(default)s]')
    # group.add_argument('--alpha_split', dest='alpha_split', default=0.95, type=float,
    #         help='alpha-split for cgm tree prior  [default: %(default)s]')   
    # group.add_argument('--beta_split', dest='beta_split', default=0.5, type=float,
    #         help='beta_split for cgm tree prior [default: %(default)s]')    
    # group.add_argument('--alpha', dest='alpha', default=1.0, type=float,
    #         help='alpha denotes the concentration of dirichlet parameter'\
    #         ' (NOTE: each of K classes will have mass alpha/K) [default: %(default)s]')
    # # kappa_0 < 1 implies that the prior mean can exhibit higher variance around the empirical mean (different means  in different partitions)
    # group.add_argument('--alpha_0', dest='alpha_0', default=2.0, type=float,
    #         help='alpha_0 is parameter of Normal-Gamma prior')
    # group.add_argument('--beta_0', dest='beta_0', default=1.0, type=float,
    #         help='beta_0 is parameter of Normal-Gamma prior')
    # group.add_argument('--mu_0', dest='mu_0', default=0.0, type=float,
    #         help='mu_0 is parameter of Normal-Gamma prior')
    # group.add_argument('--kappa_0', dest='kappa_0', default=0.3, type=float,   
    #         help='kappa_0 is parameter of Normal-Gamma prior')
    # group.add_argument('--alpha_bart', dest='alpha_bart', default=3.0, type=float,
    #         help='alpha_bart is the df parameter in BART')  # they try just 3 and 10
    # group.add_argument('--k_bart', dest='k_bart', default=2.0, type=float,
    #         help='k_bart controls the prior over mu (mu_prec) in BART')
    # group.add_argument('--q_bart', dest='q_bart', default=0.9, type=float,
    #         help='q_bart controls the prior over sigma^2 in BART')
    return parser

def parser_add_hmc_options(parser):
    # TODO: add checks for these options
    groupHMC = parser.add_argument_group("HMC options")
    groupHMC.add_argument('--structure', nargs='+', dest='tree_struct', default=-1, type=int,
            help='proposed tree structure (0=internal node;-1=leaf/terminal node')
    return parser

def parser_check_common_options(parser, settings):
    """Checks command-line options are valid."""
    fail(parser, not(settings.save==0 or settings.save==1), 'save needs to be 0/1')
    fail(parser, not(settings.optype=='real' or settings.optype=='class'), 'optype needs to be real/class')
    # fail(parser, not(settings.prior=='cgm' or settings.prior=='bart'), 'prior needs to be cgm/bart')
    # fail(parser, not(settings.tree_prior=='cgm'), 'tree_prior needs to be cgm (mondrian yet to be implemented)')

def fail(parser, condition, msg):
    """Prints fail message for invalid command-line options.""" 
    if condition:
        print(msg)
        parser.print_help()
        sys.exit(1)

def process_command_line():
    parser = parser_add_common_options()
    parser = parser_add_hmc_options(parser)
    args = parser.parse_args()
    parser_check_common_options(parser, args)
    return args

################## Dataset related functions ##################
def normalise_data(data):
    data['x_train'] = (data['x_train'] - jnp.min(data['x_train'],axis=0)) / ( jnp.max(data['x_train'],axis=0) - jnp.min(data['x_train'],axis=0) )
    return data

def convert_data(data): 
    # converts data to jnp.array type TODO include testing data
    if not isinstance(data['x_train'], DeviceArray):
        data['x_train'] = jnp.asarray(data['x_train'])
    if not isinstance(data['y_train'], DeviceArray):
        data['y_train'] = jnp.asarray(data['y_train'])
    return data

def process_dataset(data):
    data = normalise_data(data)
    data = convert_data(data)
    return data

def summary_plot(mcmc,data,tree):
    samples = mcmc.get_samples()

    fig = plt.figure(figsize=(15,10))
    grid = fig.add_gridspec(1, 3, hspace=0.2, wspace=0.2)
    
    if(data['nx'] == 2):
        ax = fig.add_subplot(grid[:,1], projection='3d')
        plot_dataset(data,ax)
    else:
        data_grid = grid[1].subgridspec(data['nx'],1)
        axes = data_grid.subplots()
        plot_dataset(data,axes)

    samples_grid = grid[0].subgridspec(len(samples), 1)
    axes = samples_grid.subplots()
    for i,key in enumerate(samples.keys()):
        axes[i].plot(samples[key])
        axes[i].set_title(key)

    tree_grid = grid[2].subgridspec(mcmc.num_chains, 1)
    axes = tree_grid.subplots()
    for chain in range(mcmc.num_chains):
        m, c = mode(samples["index"][chain*mcmc.num_samples:(chain+1)*mcmc.num_samples]) # find mode and counts
        tree_vars = {'index':m,'tau':np.mean(samples["tau"][chain*mcmc.num_samples:(chain+1)*mcmc.num_samples],axis=0)}
        if tree.optype == 'real':
            tree_vars['mu'] = np.mean(samples['mu'][chain*mcmc.num_samples:(chain+1)*mcmc.num_samples],axis=0)
            tree_vars['sigma'] = np.mean(samples['sigma'][chain*mcmc.num_samples:(chain+1)*mcmc.num_samples],axis=0)
        if(mcmc.num_chains > 1):
            tree.draw_tree(tree_vars=tree_vars,ax=axes[chain])
        else:
            tree.draw_tree(tree_vars=tree_vars,ax=axes)
    
    plt.tight_layout()
    plt.show()

def check_dataset(settings):
    classification_datasets = set(['bc-wisc', 'spambase', 'letter-recognition', 'shuttle', 'covtype', \
            'pendigits', 'arcene', 'gisette', 'madelon', 'iris', 'magic04', 'glass','toy-class', \
            'toy-class-noise','toy-class-gauss','toy-non-sym'])
    regression_datasets = set(['toy-reg','wu'])
    if not (settings.dataset[:3] == 'toy' or settings.dataset[:4] == 'test'):
        try:
            if settings.optype == 'class':
                assert(settings.dataset in classification_datasets)
            else:
                assert(settings.dataset in regression_datasets)
        except AssertionError:
            print('Invalid dataset for optype; dataset = %s, optype = %s' % \
                    (settings.dataset, settings.optype))
            raise AssertionError

def load_data(settings):
    data = {}
    check_dataset(settings)
    if (settings.dataset == 'bc-wisc') or (settings.dataset == 'spambase') \
            or (settings.dataset == 'letter-recognition') or (settings.dataset == 'shuttle') \
            or (settings.dataset == 'covtype') or (settings.dataset == 'pendigits') \
            or (settings.dataset == 'arcene') or (settings.dataset == 'gisette') \
            or (settings.dataset == 'madelon') \
            or (settings.dataset == 'iris') or (settings.dataset == 'magic04'):
        data = pickle.load(open(settings.data_path + settings.dataset + '/' + settings.dataset + '.p', \
            "rb"))
    elif settings.dataset == 'toy':
        data = load_toy_data()
    elif settings.dataset == 'toy-class':
        data = load_toy_class_data()
    elif settings.dataset == 'toy-class-noise':
        data = load_toy_class_noise_data()
    elif settings.dataset == 'toy-class-gauss':
        data = load_toy_gauss_class_data()
    elif settings.dataset == 'toy-non-sym':
        data = load_toy_non_sym()
    elif settings.dataset == 'toy2':
        data = load_toy_data2()
    elif settings.dataset == 'toy-spam':
        data = load_toy_spam_data(20)
    elif settings.dataset[:8] == 'toy-spam':
        n_points = int(settings.dataset[9:])
        data = load_toy_spam_data(n_points)
    elif settings.dataset == 'toy-reg':
        data = load_toy_reg()
    elif settings.dataset == 'test-1':
        data = load_test_dataset_1()
    elif settings.dataset == 'wu':
        data = gen_wu_dataset()
    else:
        print('Unknown dataset: ' + settings.dataset)
        raise Exception
    assert(not data['is_sparse'])
    return data

def load_toy_data2():       
    """ easier than toy_data: 1 d marginal gives away the best split """
    n_dim = 2
    n_train_pc = 20
    n_class = 2
    n_train = n_train_pc * n_class
    n_test = n_train
    y_train = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    y_test = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    mag = 5
    x_train = np.random.randn(n_train, n_dim)
    x_test = np.random.randn(n_train, n_dim)
    for i, y_ in enumerate(y_train):
        x_train[i, :] += (2 * y_ - 1) * mag
    for i, y_ in enumerate(y_test):
        x_test[i, :] += (2 * y_ - 1) * mag
    x_train = np.round(x_train)
    x_test = np.round(x_test)
    data = {'x_train': jnp.array(x_train), 'y_train': jnp.array(y_train), 'n_class': n_class, \
            'nx': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data

def load_toy_non_sym():
    """ Classification toy dataset which is not symmetric. """
    ###TODO testing dataset
    tau1 = 0.2
    tau2 = 0.7
    indx1 = 0
    indx2 = 1
    n_dim = 2 
    n_train = 100
    n_class = 2
    n_test = n_train
    y_train = np.r_[np.ones(int(n_train/2), dtype='int'), \
            np.zeros(int(n_train/2), dtype='int')]
    y_test = np.r_[np.ones(n_train, dtype='int'), \
            np.zeros(n_train, dtype='int')]
    rand_mag = 1.
    x_train = rand_mag * np.random.randn(n_train, n_dim)
    x_test = rand_mag * np.random.randn(n_train, n_dim)
    mag = 3

    for i, y_ in enumerate(y_train):
        if y_ == 0:
            x_train[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_train[i, :] += np.array([tmp, -tmp]) * mag

    indx = x_train[:,1] <= 0
    x_train[indx,0] += tau1*mag
    x_train[~indx,0] += tau2*mag

    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'nx': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data

def load_toy_data():
    nx = 2
    n_train_pc = 20
    ny = 2
    n_train = n_train_pc * ny
    n_test = n_train
    y_train = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    y_test = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    x_train = np.random.randn(n_train, nx)
    x_test = np.random.randn(n_train, nx)
    mag = 5
    for i, y_ in enumerate(y_train):
        if y_ == 0:
            x_train[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_train[i, :] += np.array([tmp, -tmp]) * mag
    for i, y_ in enumerate(y_test):
        if y_ == 0:
            x_test[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_test[i, :] += np.array([tmp, -tmp]) * mag
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': ny, \
            'nx': nx, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data
 
def load_test_dataset_1():
    n_dim = 2 
    n_train_pc = 1
    n_class = 2
    n_train = n_train_pc * n_class
    n_test = n_train
    y_train = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    y_test = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    x_train = np.random.randn(n_train, n_dim)
    x_test = np.random.randn(n_train, n_dim)
    mag = 5
    for i, y_ in enumerate(y_train):
        if y_ == 0:
            x_train[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_train[i, :] += np.array([tmp, -tmp]) * mag
    for i, y_ in enumerate(y_test):
        if y_ == 0:
            x_test[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_test[i, :] += np.array([tmp, -tmp]) * mag
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'nx': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    print(data)
    return data
   
def gen_chipman_reg(n_points):
    n_dim = 2
    x = np.zeros((n_points, n_dim))
    y = np.zeros(n_points)
    f = np.zeros(n_points)
    x[:, 0] = [math.ceil(x_) for x_ in np.random.rand(n_points)*10]
    x[:, 1] = [math.ceil(x_) for x_ in np.random.rand(n_points)*4]
    for i, x_ in enumerate(x):
        if x_[1] <= 4.0:
            if x_[0] <= 5.0:
                f[i] = 8.0
            else:
                f[i] = 2.0
        else:
            if x_[0] <= 3.0:
                f[i] = 1.0
            elif x_[0] > 7.0:
                f[i] = 8.0
            else:
                f[i] = 5.0
    y = f + 0.02 * np.random.randn(n_points)
    return x, y

def load_toy_reg():
    n_dim = 2
    n_train = 200
    n_test = n_train
    x_train, y_train = gen_chipman_reg(n_train)
    x_test, y_test = gen_chipman_reg(n_test)
    data = {'x_train': x_train, 'y_train': y_train, \
            'nx': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data

def load_toy_spam_data(dim):
    n_dim = dim
    n_dim_rel = 2   # number of relevant dimensions
    n_train_pc = 100  
    n_class = 2
    n_train = n_train_pc * n_class
    n_test = n_train
    y_train = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    y_test = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    x_train = np.random.randn(n_train, n_dim)
    x_test = np.random.randn(n_train, n_dim)
    mag = 3 
    for i, y_ in enumerate(y_train):
        if y_ == 0:
            x_train[i, :n_dim_rel] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_train[i, :n_dim_rel] += np.array([tmp, -tmp]) * mag
    for i, y_ in enumerate(y_test):
        if y_ == 0:
            x_test[i, :n_dim_rel] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_test[i, :n_dim_rel] += np.array([tmp, -tmp]) * mag
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data

def gen_wu_dataset():
    """ Generates the synthetic data as described in Section 5.1 of (Wu, et al. 2007). """
    # TODO: generate test part of data set 
    # TODO: make faster

    p = 3 # number of predictors
    n = 300 # number of observations

    x_train = np.zeros([n,p])
    y_train = np.zeros(n)

    for i in range(n):
        if i < 200:
            x_train[i,0] = np.random.uniform(0.1,0.4)
            x_train[i,2] = np.random.uniform(0.6,0.9)
            if i < 100:
                x_train[i,1] = np.random.uniform(0.1,0.4)
            else:
                x_train[i,1] = np.random.uniform(0.6,0.9)
        else:
            x_train[i,0] = np.random.uniform(0.6,0.9)
            x_train[i,1] = np.random.uniform(0.1,0.9)
            x_train[i,2] = np.random.uniform(0.1,0.4)   
            
            
        if (x_train[i,0] <= 0.5)&(x_train[i,1] <= 0.5):
            y_train[i] = 1 + np.random.normal(0,0.25)
        elif (x_train[i,0] <= 0.5)&(x_train[i,1] > 0.5):
            y_train[i] = 3 + np.random.normal(0,0.25)
        else:
            y_train[i] = 5 + np.random.normal(0,0.25)
    
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': 3, \
            'nx': p, 'n_train': n, 'x_test': [], \
            'y_test': [], 'n_test': 0, 'is_sparse': False}
    return data

def load_toy_class_data():
    # TODO make training values
    p = 3 # number of predictors
    n = 100 # number of observations
    x_train = np.zeros([n,p])
    y_train = np.zeros(n)
    eps = 0.0 # deadzone region
    tau_true = 0.5

    x_train[:,0] = np.random.uniform(0,1,n) # x0 is random, does not effect output
    x_train[:,2] = np.random.uniform(0,1,n) # x2 is random, does not effect output
    indx = np.random.uniform(size=n) < tau_true
    x_train[indx,1] = np.random.uniform(0,tau_true-eps,size=sum(indx))
    x_train[~indx,1] = np.random.uniform(tau_true+eps,1,size=n-sum(indx))
    indx = x_train[:,1] <= tau_true
    y_train[indx] = 0
    y_train[~indx] = 1

    data = {'x_train': x_train, 'y_train': y_train, 'n_class': 2, \
            'nx': p, 'n_train': n, 'x_test': [], 'y_test': [], \
            'n_test': 0, 'theta_true': {'i':1, 'tau':tau_true}, 'is_sparse': False}
    return data

def load_toy_class_noise_data():
    data = load_toy_class_data()
    # Create noisy dataset - randomly swap values of some indicies
    indx = random.sample(range(0,data['n_train']), int(np.floor(data['n_train']/10)))
    data['y_train']= np.array(data['y_train'])
    data['y_train'][indx] = 1 - data['y_train'][indx]
    data['y_train'] = data['y_train']
    return data

def load_toy_gauss_class_data():
    """ Create dataset - Gaussian distributed x values, y in {0,1}. """
    # TODO make training values
    p = 1 # number of predictors
    n = 100 # number of observations
    x_train = np.zeros([n,p])
    y_train = np.zeros(n)
    tau_true = 0.5
    indx = np.random.uniform(size=n) < tau_true
    x_train[indx] = np.random.normal(0.25,0.08,size=(sum(indx),1))
    y_train[indx] = 0
    x_train[~indx] = np.random.normal(0.75,0.08,size=(n-sum(indx),1))
    y_train[~indx] = 1
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': 2, \
        'nx': p, 'n_train': n, 'x_test': [], 'y_test': [], \
        'n_test': 0, 'theta_true': tau_true, 'is_sparse': False}
    return data

def plot_dataset(data, ax=None, plot_now=False):
    if(data['nx'] == 1): # one predictor variable 
        if(ax is None):
            plot_now = True
            plt.figure(figsize=(15,10))  
        plt.plot(data['x_train'],data['y_train'],'*')
        plt.xlabel("x")
        plt.ylabel("y")
    elif(data['nx'] == 2): # two predictor variables
        if(ax is None):
            plot_now = True
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data['x_train'][:,0],data['x_train'][:,1],data['y_train'])
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('y')
    else: # otherwise, visualise separately
        if(ax is None):
            plot_now = True
            fig, ax = plt.subplots(1,data['nx'],figsize=(15,10)) 
        # else: # TODO
        #     fig, ax = plt.subplots(2,2,figsize=(15,10)) 
        for i in range(data['nx']):
            ax[i].plot(np.array(data['x_train'])[:,i],data['y_train'],'*')
            ax[i].set_xlabel("x"+str(i)) ### TODO CHANGE BACK TO i+1
            ax[i].set_ylabel("y")

    if(plot_now == True):
        plt.show()



################## Tree Related Functions ##################
def get_parent_id(node_id):
    if node_id == 0:
        op = None
    else:
        op = int(math.ceil(node_id / 2.) - 1)
    return op

def get_children_id(node_id):
    tmp = 2 * (node_id + 1)
    return (tmp - 1, tmp)

def get_depth(node_id):
    op = int(math.floor(math.log(node_id + 1, 2)))
    return op

def get_node_list(depth):
    if depth == 0:
        op = [0]
    else:
        op = [2 ** depth - 1 + x for x in range(2 ** depth)]
    return op

def get_no_grandchildren(tree):
    no_gc = []
    for node in tree.internal_nodes:
        children = get_children_id(node)
        if(children[0] in range(tree.num_nodes)): # node had children
            if((get_children_id(children[0])[0] in range(tree.num_nodes)) or (get_children_id(children[1])[0] in range(tree.num_nodes))):
                pass
            else:
                no_gc.append(node)
    return no_gc

def traverse(tree,x):
    node = 0
    while True:
        if node in tree.leaf_nodes:
            break
        left, right = get_children_id(node)
        var, cp = tree.node_info[node]
        if (x[var] <= cp):
            node = left
        else:
            node = right
    return node

################## General Functions ##################
def hist_count(x, basis):
    """
    counts number of times each element in basis appears in x
    op is a vector of same size as basis
    assume no duplicates in basis
    """
    op = jnp.zeros((len(basis)), dtype=int)
    for i in range(len(op)):
        op = ops.index_update(op,ops.index[i],jnp.sum(x == basis[i]))
    return op

def logsumexp(x):
    tmp = x.copy()
    tmp_max = np.max(tmp)
    tmp -= tmp_max
    op = np.log(np.sum(np.exp(tmp))) + tmp_max
    return op

def softmax(x):
    tmp = x.copy()
    tmp_max = np.max(tmp)
    tmp -= float(tmp_max)
    tmp = np.exp(tmp)
    op = tmp / np.sum(tmp)
    return op

def assert_no_nan(mat, name='matrix'):
    try:
        assert(not any(np.isnan(mat)))
    except AssertionError:
        print('%s contains NaN' % name)
        print(mat)
        raise AssertionError

def check_if_one(val):
    try:
        assert(np.abs(val - 1) < 1e-12)
    except AssertionError:
        print('val = %s (needs to be equal to 1)' % val)
        raise AssertionError

def check_if_zero(val):
    try:
        assert(np.abs(val) < 1e-10)
    except AssertionError:
        print('val = %s (needs to be equal to 0)' % val)
        raise AssertionError

def sample_multinomial(prob):
    try:
        k = int(np.where(np.random.multinomial(1, prob, size=1)[0]==1)[0])
    except TypeError:
        print('problem in sample_multinomial: prob = ')
        print(prob)
        raise TypeError
    except:
        raise Exception
    return k

def sample_multinomial_scores(scores):
    scores_cumsum = np.cumsum(scores)
    s = scores_cumsum[-1] * np.random.rand(1)
    k = 0
    while s > scores_cumsum[k]:
        k += 1
    return k

def sample_polya(alpha_vec, n):
    """ alpha_vec is the parameter of the Dirichlet distribution, n is the #samples """
    prob = np.random.dirichlet(alpha_vec)
    n_vec = np.random.multinomial(n, prob)
    return n_vec

def get_kth_minimum(x, k=1):
    """ gets the k^th minimum element of the list x 
        (note: k=1 is the minimum, k=2 is 2nd minimum) ...
        based on the incomplete selection sort pseudocode """
    n = len(x)
    for i in range(n):
        minIndex = i
        minValue = x[i]
        for j in range(i+1, n):
            if x[j] < minValue:
                minIndex = j
                minValue = x[j]
        x[i], x[minIndex] = x[minIndex], x[i]
    return x[k-1]

class empty(object):
    def __init__(self):
        pass

