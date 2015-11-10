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
        self.infected_nodes = set([])
        self.edge_set = set([])
        super(TemporalGraph, self).__init__()  #Calls parent constructor
    
    def _fix(self, layout = None):
        '''Fixes the position of the nodes, so each subsequent plot will
        have same node positions.'''
        if layout == None:
            self.pos = self.layout(self)
        else:
            self.pos = layout(self)
        #
    
    def add_edges_from(self, _iterable):
        # Make sure we're dealing with a set of edges
        _set = set(_iterable)
        # Determine which nodes to remove and add
        edges_to_remove = self.edge_set - _set
        super(TemporalGraph, self).remove_edges_from(edges_to_remove)
        edges_to_add = _set - self.edge_set
        super(TemporalGraph, self).add_edges_from(edges_to_add)
        # Update nodes in the network
        self.edge_set = _set
    
    def add_edge(self, edge):
        self.add_edges_from(set([edge]))
    
    def draw(self, filename = None, show = True, **kwargs):
        # Construct a layout if one doesn't exist already
        if self.pos == None:
            self.pos = nx.spring_layout(self)
        
        # Draw the uninfected nodes
        pure_nodes = set(self.nodes()) - set(self.infected_nodes)
        nx.draw(self, pos = self.pos, nodelist = pure_nodes,
                node_color = state2col['pure'], **kwargs)
        # Draw the infected nodes
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
    
    def update_position(self, pos = None):
        '''Updates node position. If no position dict is provided, generates
        a spring layout.'''
        if pos == None:
            self.pos = nx.spring_layout(self)
        else:
            self.pos = pos
    
    def run_step(self):
        '''Runs a single step of a simulation, i.e. giving every node a chance
        to infect its neighbors.'''

        # Queue actions to delay them until every node has had its turn
        queue = deque([])
        for node in self.infected_nodes:
            for neighbor in self.neighbors_iter(node):
                # Infect them
                queue.append((self.infect, neighbor))
            #
        # All actions queued - excecute job queue.
        while queue:
            f, args = queue.pop()
            f(args)
    
    
    def run_all(self):
        pass  #TODO denne her skal lave run_step + update indtil alle dør eller timelinen slutter
    
    def components_are_homogenous(self):
        '''Determines whether the nodes in each individual component of the
        have the same status, i.e. if it's true for every component that all
        nodes are either infected or healthy.'''
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
    
    def is_extinct(self):
        return len(self) == len(self.infected_nodes)
        

def main():
    pass


if __name__ == '__main__':
    G = TemporalGraph()
    n_nodes = 100
    G.add_nodes_from(range(n_nodes))
    G.draw()
    G.infect_random_node()
    counter = 1
    lim = float('inf')
    
    while not G.is_extinct() and counter < lim:
        new_edges = [(np.random.randint(n_nodes), np.random.randint(n_nodes))
                     for _ in xrange(10)]
        G.add_edges_from(new_edges)
        G.draw(node_size = 50, show = False, filename = 'temp/f%03da' % counter)
        G.run_step()
        G.draw(node_size = 50, show = False, filename = 'temp/f%03db' % counter)
        counter += 1
        G.run_step()
        print counter
