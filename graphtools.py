# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 15:59:31 2015

@author: Bjarke
"""

from __future__ import division
from collections import Counter, deque
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
import time

state2col = {'infected' : 'red', 'pure' : 'green'}

class TemporalGraph(nx.Graph):
    '''Graph class which makes temporal dynamics easier to model.'''
    def __init__(self):
        self.pos = None  # Position dict to ensure consistent node placement
        self.queue = deque([])  # todo-list for next time step update
        self.layout = nx.spring_layout
        self.infected_nodes = set([])
        super(TemporalGraph, self).__init__()  #Calls parent constructor
    
    def _fix(self, layout = None):
        '''Fixes the position of the nodes, so each subsequent plot will
        have same node positions.'''
        if layout == None:
            self.pos = self.layout(self)
        else:
            self.pos = layout(self)
        #
    
    # We just override these to make sure node position is fixed
    def add_edges_from(self, _iterable):
        super(TemporalGraph, self).add_edges_from(_iterable)
        self._fix()    
    def add_nodes_from(self, _iterable):
        super(TemporalGraph, self).add_nodes_from(_iterable)
        self._fix()
    #TODO single node. For doven nu...
    
    def draw(self, filename = None, show = True, **kwargs):
        pure_nodes = set(self.nodes()) - set(self.infected_nodes)
        nx.draw(self, pos = self.pos, nodelist = pure_nodes,
                node_color = state2col['pure'], **kwargs)
        nx.draw(self, pos = self.pos, nodelist = self.infected_nodes,
                node_color = state2col['infected'], **kwargs)
        if filename:
            plt.savefig(filename)
        if show:
            plt.show()
        plt.clf()
    
    def infect(self, node):
        '''Infects target node'''
        self.infected_nodes.add(node)
    
    def infect_random_node(self):
        node = np.random.choice(self.nodes())
        self.infect(node)
    
    def run(self):
        for node in self.infected_nodes:
            for neighbor in self.neighbors_iter(node):
                # Infect them
                if np.random.uniform(0,1) < 1:
                    self.queue.append((self.infect, neighbor))
                #
            #
        self.update()
    
    def update(self):
        while self.queue:
            f, args = self.queue.pop()
            f(args)
        #
    
    def is_extinct(self):
        components = nx.connected_component_subgraphs(self)
        for component in components:
            nodes = component.nodes()
            if len(nodes) == 1:
                continue
            for i in xrange(1, len(nodes)):
                if (nodes[i-1] in self.infected_nodes) != (nodes[i] in self.infected_nodes):
                    return False
                #
            #
        return True
        

def main():
    G = TemporalGraph()
    G.add_edges_from([(np.random.randint(0, 600), np.random.randint(0, 600))
                     for _ in xrange(1000)])
    G.infect_random_node()
    counter = 1
    while not G.is_extinct():
        print len(G.infected_nodes)
        G.draw(node_size = 50, show = False, filename = 'frames/f%03d' % counter)
        counter += 1
        G.run()
        G.update()


if __name__ == '__main__':
    main()