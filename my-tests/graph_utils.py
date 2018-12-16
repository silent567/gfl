#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def random_graph(size=None,sparse_rate=0.1):
    if size is None:
        size = np.random.randint(1,50)
    A = np.random.uniform(0,3,[size,size])
    A *= np.random.uniform(0,1,[size,size])<sparse_rate
    A = np.triu(A)
    A *= 1-np.eye(size)
    A += np.transpose(A)
    return A

def random_node_weights(size):
    return 10*np.random.uniform(0,1,size)

def random_graph_node_weights(size=None,sparse_rate=0.1):
    A = random_graph(size,sparse_rate)
    nw = random_node_weights(A.shape[1])
    return A,nw

def draw_graph_size(A,node_weights,pos=nx.circular_layout,vmin=0,vmax=1):
    if vmin is not None and vmax is not None:
        node_weights = 300*(node_weights-vmin)/(vmax-vmin)
    G = nx.from_numpy_matrix(A)
    nx.draw_networkx(G,node_size=node_weights,pos=pos(G),cmap='Greys',vmin=vmin,vmax=vmax,with_labels=False,width=[v['weight'] for _,_,v in G.edges(data=True)])

def draw_graph_color(A,node_weights,pos=nx.circular_layout,vmin=0,vmax=1):
    G = nx.from_numpy_matrix(A)
    nx.draw_networkx(G,node_color=node_weights,pos=pos(G),cmap='Greys',vmin=vmin,vmax=vmax,with_labels=False,width=[v['weight'] for _,_,v in G.edges(data=True)])

def draw_graph_size_save(A,node_weights,output_file='test.png',pos=nx.circular_layout,vmin=0,vmax=1):
    plt.close()
    draw_graph_size(A,node_weights,pos,vmin,vmax)
    plt.savefig(output_file)

def draw_graph_color_save(A,node_weights,output_file='test.png',pos=nx.circular_layout,vmin=0,vmax=1):
    plt.close()
    draw_graph_color(A,node_weights,pos,vmin,vmax)
    plt.savefig(output_file)

