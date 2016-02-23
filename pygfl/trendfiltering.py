'''Copyright (C) 2016 by Wesley Tansey

    This file is part of the GFL library.

    The GFL library is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    The GFL library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with the GFL library.  If not, see <http://www.gnu.org/licenses/>.
'''
import matplotlib.pylab as plt
import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.sparse import coo_matrix
from ctypes import *
from utils import *

'''Load the graph trend filtering library'''
graphfl_lib = cdll.LoadLibrary('libgraphfl.so')
weighted_graphtf = graphfl_lib.graph_trend_filtering_weight_warm
weighted_graphtf.restype = c_int
weighted_graphtf.argtypes = [c_int, ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'), c_double,
                    c_int, c_int, c_int,
                    ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                    c_int, c_double,
                    ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

weighted_graphtf_logit = graphfl_lib.graph_trend_filtering_logit_warm
weighted_graphtf_logit.restype = c_int
weighted_graphtf_logit.argtypes = [c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), c_double,
                    c_int, c_int, c_int,
                    ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                    c_int, c_double,
                    ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

class TrendFilteringSolver:
    def __init__(self, maxsteps=3000, converge=1e-6):
        self.maxsteps = maxsteps
        self.converge = converge

    def set_data(self, y, D, weights=None):
        self.y = y
        self.nnodes = len(y)
        self.D = D
        self.weights = weights
        self.beta = np.zeros(self.nnodes, dtype='double')
        self.steps = []
        self.Dk = None
        self.u = None

    def solve(self, k, lam):
        '''Solves the GFL for a fixed value of lambda.'''
        if self.Dk is None:
            self.Dk = get_delta(self.D, k).tocoo()
        if self.u is None:
            self.u = np.zeros(self.Dk.shape[0], dtype='double')
        if self.weights is None:
            self.weights = np.ones(len(self.y), dtype='double')
        s = weighted_graphtf(self.nnodes, self.y, self.weights, lam,
                             self.Dk.shape[0], self.Dk.shape[1], self.Dk.nnz,
                             self.Dk.row.astype('int32'), self.Dk.col.astype('int32'), self.Dk.data.astype('double'),
                             self.maxsteps, self.converge,
                             self.beta, self.u)
        self.steps.append(s)
        return self.beta

class LogitTrendFilteringSolver:
    def __init__(self, maxsteps=3000, converge=1e-6):
        self.maxsteps = maxsteps
        self.converge = converge

    def set_data(self, trials, successes, D):
        self.trials = trials
        self.successes = successes
        self.nnodes = len(trials)
        self.D = D
        self.beta = np.zeros(self.nnodes, dtype='double')
        self.steps = []
        self.Dk = None
        self.u = None

    def solve(self, k, lam):
        '''Solves the GFL for a fixed value of lambda.'''
        if self.Dk is None:
            self.Dk = get_delta(self.D, k).tocoo()
        if self.u is None:
            self.u = np.zeros(self.Dk.shape[0], dtype='double')
        s = weighted_graphtf_logit(self.nnodes, self.trials, self.successes, lam,
                                 self.Dk.shape[0], self.Dk.shape[1], self.Dk.nnz,
                                 self.Dk.row.astype('int32'), self.Dk.col.astype('int32'), self.Dk.data.astype('double'),
                                 self.maxsteps, self.converge,
                                 self.beta, self.u)
        self.steps.append(s)
        return self.beta


def test_solve_gtf():
    # Load the data and create the penalty matrix
    max_k = 3
    y = (np.sin(np.linspace(-np.pi, np.pi, 100)) + 1) * 5
    y[25:75] += np.sin(np.linspace(1.5*-np.pi, np.pi*2, 50))*5 ** (np.abs(np.arange(50) / 25.))
    y += np.random.normal(0,1.0,size=len(y))
    # np.savetxt('/Users/wesley/temp/tfdata.csv', y, delimiter=',')
    # y = np.loadtxt('/Users/wesley/temp/tfdata.csv', delimiter=',')
    mean_offset = y.mean()
    y -= mean_offset
    stdev_offset = y.std()
    y /= stdev_offset
    
    # equally weight each data point
    w = np.ones(len(y))

    lam = 3.

    # try different weights for each data point
    # w = np.ones(len(y))
    # w[0:len(y)/2] = 1.
    # w[len(y)/2:] = 100.
    
    D = coo_matrix(get_1d_penalty_matrix(len(y)))

    z = np.zeros((max_k,len(y)))
    for k in xrange(max_k):
        tf = TrendFilteringSolver()
        tf.set_data(y, D, w)
        z[k] = tf.solve(k, lam)
    
    y *= stdev_offset
    y += mean_offset
    z *= stdev_offset
    z += mean_offset


    colors = ['orange', 'skyblue', '#009E73', 'purple']
    fig, ax = plt.subplots(max_k)
    x = np.linspace(0,1,len(y))
    for k in xrange(max_k):
        ax[k].scatter(x, y, alpha=0.5)
        ax[k].plot(x, z[k], lw=2, color=colors[k], label='k={0}'.format(k))
        ax[k].set_xlim([0,1])
        ax[k].set_ylabel('y')
        ax[k].set_title('k={0}'.format(k))
    
    plt.show()
    plt.clf()

def test_solve_gtf_logit():
    max_k = 5
    trials = np.random.randint(5, 30, size=100).astype('int32')
    probs = np.zeros(100)
    probs[:25] = 0.25
    probs[25:50] = 0.75
    probs[50:75] = 0.5
    probs[75:] = 0.1
    successes = np.array([np.random.binomial(t, p) for t,p in zip(trials, probs)]).astype('int32')

    lam = 3.

    D = coo_matrix(get_1d_penalty_matrix(len(trials)))
    z = np.zeros((max_k,len(trials)))
    for k in xrange(max_k):
        tf = LogitTrendFilteringSolver()
        tf.set_data(trials, successes, D)
        z[k] = tf.solve(k, lam)

    colors = ['orange', 'skyblue', '#009E73', 'purple', 'black']
    fig, ax = plt.subplots(max_k+1)
    x = np.linspace(0,1,len(trials))
    ax[0].bar(x, successes, width=1./len(x), color='darkblue', alpha=0.3)
    ax[0].bar(x, trials-successes, width=1./len(x), color='skyblue', alpha=0.3, bottom=successes)
    ax[0].set_ylim([0,30])
    ax[0].set_xlim([0,1])
    ax[0].set_ylabel('Trials and successes')
    for k in xrange(max_k):
        ax[k+1].scatter(x, probs, alpha=0.5)
        ax[k+1].plot(x, z[k], lw=2, color=colors[k], label='k={0}'.format(k))
        ax[k+1].set_ylim([0,1])
        ax[k+1].set_xlim([0,1])
        ax[k+1].set_ylabel('Probability of success')
    
    plt.show()
    plt.clf()

if __name__ == '__main__':
    test_solve_gtf_logit()
