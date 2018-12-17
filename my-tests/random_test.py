#!/usr/bin/env python
# coding=utf-8

from mapping import *
from graph_utils import *
import matplotlib.pyplot as plt

def draw_save(A,z,output_file=None,lam=1,gamma=1):
    fig = plt.figure(figsize=(12,12))
    # plt.axis('off')
    ax = plt.subplot(2,3,1)
    ax.axis('off')
    draw_graph_size(A,z*30,vmin=None,vmax=None)

    ax = plt.subplot(2,3,2)
    ax.axis('off')
    draw_graph_size(A,softmax(z))

    ax = plt.subplot(2,3,3)
    ax.axis('off')
    draw_graph_size(A,sparsemax(z,gamma))

    ax = plt.subplot(2,3,4)
    ax.axis('off')
    draw_graph_size(A,gfusedlasso(z,A,lam=lam)*30,vmin=None,vmax=None)

    ax = plt.subplot(2,3,5)
    ax.axis('off')
    draw_graph_size(A,gfusedmax(z,A,lam=lam,gamma=gamma))

    if output_file is None:
        output_file = 'random_%d_%d_%.2f_%.2f.png'%(z.size,np.sum(np.triu(A)>0),lam,gamma)
    plt.savefig(output_file,dpi=300)
    plt.close()

if __name__ == '__main__':
    for lam in np.arange(-2,2,0.5):
        for gamma in np.arange(-2,2,0.5):
            for _ in range(20):
                A,z = random_graph_node_weights()
                draw_save(A,z,lam=10**lam,gamma=10**gamma)
