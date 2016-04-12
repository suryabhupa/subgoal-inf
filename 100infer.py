from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import matplotlib
import pylab
import os
import sys
matplotlib.rcParams['font.size'] = 8

import pyhsmm
import models
from pyhsmm.util.stats import cov
from pyhsmm.util.text import progprint_xrange

print \
'''
This demo shows how HDP-HMMs can fail when the underlying data has state
persistence without some kind of temporal regularization (in the form of a
sticky bias or duration modeling): without setting the number of states to be
the correct number a priori, lots of extra states can be intsantiated.
BUT the effect is much more relevant on real data (when the data doesn't exactly
fit the model). Maybe this demo should use multinomial emissions...
'''

###############
#  load data  #
###############

testcase = sys.argv[1]
if testcase == "mr":
    path = 'data/' + testcase + '/frames_features.txt'
elif testcase == "mr2":
    path = 'atari/exp1/frames_features.txt'
else:
    path = 'data/' + testcase + '/Goto-' + testcase + '-feature-state-ts.txt'

# data = np.loadtxt(os.path.join(os.path.dirname(__file__),'MultiGoals-feature-state-ts.txt'))[:2500]
data = np.loadtxt(os.path.join(os.path.dirname(__file__), path))
# data = np.loadtxt(os.path.join(os.path.dirname(__file__),'LightKey-feature-state-ts.txt'))[:2500]
print(data)
mean = data.mean(axis=1)
data = data - mean[:, np.newaxis]

#########################
#  posterior inference  #
#########################

# Set the weak limit truncation level
Nmax = 10

# Set iteration count
ITERATIONS = 100

# and some hyperparameters
obs_dim = data.shape[1]

## subHMMs
Nsuper = 10
Nsub = 10
T = 1000

try:
    import brewer2mpl
    plt.set_cmap(brewer2mpl.get_map('Set1','qualitative',max(3,min(8,Nsuper))).mpl_colormap)
except:
    pass

obs_hypparams = dict(
        mu_0=np.zeros(obs_dim),
        sigma_0=np.eye(obs_dim),
        kappa_0=0.1,
        nu_0=obs_dim+10,
        )

dur_hypparams = dict(
        r_discrete_distn=np.r_[0,0,0,0,0,1.,1.,1.],
        alpha_0=40,
        beta_0=10,
        )

""""

true_obs_distnss = [[pyhsmm.distributions.Gaussian(**obs_hypparams) for substate in xrange(Nsub)]
        for superstate in xrange(Nsuper)]

true_dur_distns = [pyhsmm.distributions.NegativeBinomialIntegerR2Duration(
    **dur_hypparams) for superstate in range(Nsuper)]

truemodel = models.WeakLimitHDPHSMMSubHMMs(
        init_state_concentration=6.,
        sub_init_state_concentration=6.,
        alpha=10.,gamma=10.,
        sub_alpha=10.,sub_gamma=10.,
        obs_distnss=true_obs_distnss,
        dur_distns=true_dur_distns) 

Nmaxsuper = 2*Nsuper
Nmaxsub = 2*Nsub

obs_distnss = \
        [[pyhsmm.distributions.Gaussian(**obs_hypparams)
            for substate in range(Nmaxsub)] for superstate in range(Nmaxsuper)]

dur_distns = \
        [pyhsmm.distributions.NegativeBinomialIntegerR2Duration(
            **dur_hypparams) for superstate in range(Nmaxsuper)]

model = models.WeakLimitHDPHSMMSubHMMs(
        init_state_concentration=6.,
        sub_init_state_concentration=6.,
        alpha=6.,gamma=6.,
        sub_alpha=6.,sub_gamma=6.,
        obs_distnss=obs_distnss,
        dur_distns=dur_distns)

model.add_data(data)
model.resample_parameters()
model.resample_parameters()
model.resample_parameters()

###############
#  inference  #
###############

for itr in progprint_xrange(ITERATIONS):
    model.resample_model()

plt.figure()
model.plot()
plt.gcf().suptitle('subHSMM sampled model after {} iterations'.format(ITERATIONS))
plt.savefig('plots/' + testcase + '/subhmm.png')
plt.close()
s = model.states_list[0] 
"""

### HDP-HMM without the sticky bias

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(Nmax)]
posteriormodel = pyhsmm.models.WeakLimitHDPHMM(alpha=6.,gamma=6.,init_state_concentration=1.,
                                   obs_distns=obs_distns)
posteriormodel.add_data(data)

for idx in progprint_xrange(ITERATIONS):
    posteriormodel.resample_model()

posteriormodel.plot()
plt.gcf().suptitle('HDP-HMM sampled model after {} iterations'.format(ITERATIONS))
plt.savefig('plots/' + testcase + '/100-hdp-hmm.png')
plt.close() 

# Some more hypparams

obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.3,
                'nu_0':obs_dim+5}
dur_hypparams = {'alpha_0':2,
                 'beta_0':2}

### Sticky-HDP-HMM

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(Nmax)]
posteriormodel = pyhsmm.models.WeakLimitStickyHDPHMM(
        kappa=50.,alpha=6.,gamma=6.,init_state_concentration=1.,
        obs_distns=obs_distns)
posteriormodel.add_data(data)

for idx in progprint_xrange(ITERATIONS):
    posteriormodel.resample_model()

posteriormodel.plot()
plt.gcf().suptitle('Sticky HDP-HMM sampled model after {} iterations'.format(ITERATIONS))
plt.savefig('plots/' + testcase + '/100-sticky-hdp-hmm.png')
plt.close()

'''
This demo shows the HDP-HSMM in action. Its iterations are slower than those for
the (Sticky-)HDP-HMM, but explicit duration modeling can be a big advantage for 
conditioning the prior or for discovering structure in data. 
'''

## HDP-HSMM

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        alpha=6.,gamma=6., # these can matter; see concentration-resampling.py
        init_state_concentration=6., # pretty inconsequential
        obs_distns=obs_distns,
        dur_distns=dur_distns)
posteriormodel.add_data(data,trunc=60) # duration truncation speeds things up when it's possible

for idx in progprint_xrange(ITERATIONS):
    posteriormodel.resample_model()

posteriormodel.plot()
plt.gcf().suptitle('HDP-HSMM sampled model after {} iterations'.format(ITERATIONS))
plt.savefig('plots/' + testcase + '/100-hdp-hsmm.png')
plt.close() 


plt.show()
