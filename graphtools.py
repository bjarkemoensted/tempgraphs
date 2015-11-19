# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 15:59:31 2015

@author: Bjarke
"""

from __future__ import division
from collections import Counter, deque
import itertools
import json
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
        self.timeline = None
        self._timeline_pointer = None  #Tracks progress in timeline
        self._velocity = None
        self._edgelist_queue = deque([])
        self._timescale = 48  #Number of frames between hard position updates
        super(TemporalGraph, self).__init__()  #Calls parent constructor
    
    def add_edges_from(self, _iterable):
        # Make sure we're dealing with a set of edges
        try:
            _set = set(_iterable)
        except TypeError:
            _set = set([tuple(edge) for edge in _iterable])
        # Determine which nodes to remove and add
        edges_to_remove = self.edge_set - _set
        self.remove_edges_from(edges_to_remove)
        edges_to_add = _set - self.edge_set
        super(TemporalGraph, self).add_edges_from(edges_to_add)
        # Update nodes in the network
        self.edge_set = _set
    
    def add_edge(self, edge):
        self.add_edges_from(set([edge]))
    
    def add_nodes_from(self, _iterable):
        _set = set(_iterable)
        current = set(self.nodes())
        nodes_to_remove = current - _set
        self.remove_nodes_from(nodes_to_remove)
        nodes_to_add = _set - current
        super(TemporalGraph, self).add_nodes_from(nodes_to_add)
        # Just assign random positions to the new nodes
        if self.pos:
            for node in nodes_to_add:
                x = np.random.uniform(0,1)
                y = np.random.uniform(0,1)
                self.pos[node] = (x,y)
    
    def add_timeline(self, timeline):
        self.timeline = timeline
        self._timeline_pointer = 0
        
            
    def update(self):
        if not self.timeline:
            raise ValueError('Missing timeline')
            
        if not self._edgelist_queue:
            
            # Update edgelist queue
            temp = nx.Graph()
            while len(self._edgelist_queue) < self._timescale:
                element = self.timeline[self._timeline_pointer]
                if isinstance(element, dict):
                    edgelist = element['edgelist']
                elif isinstance(element, list):
                    edgelist = element
                else:
                    raise TypeError(type(element))
                
                temp.add_edges_from(edgelist)
                self._edgelist_queue.append(edgelist)
                
                self._timeline_pointer += 1
            
            # Add nodes encountered to TempGraph.
            self.add_nodes_from(temp.nodes())
            
            # If no position is set yet, add circular layout
            if not self.pos:
                self.pos = nx.circular_layout(temp)

            # Find the position to move toward during the next frames
            next_pos = nx.spring_layout(temp)
            
            normfac = 1.0/self._timescale
            self._velocity = {}
            for node, newpos in next_pos.iteritems():
                oldpos = self.pos[node]
                dx = (newpos[0] - oldpos[0])*normfac
                dy = (newpos[1] - oldpos[1])*normfac
                self._velocity[node] = (dx, dy)
            #
            del temp
        # Now updte edges and position - first edges
        edgelist = self._edgelist_queue.pop()
        self.add_edges_from(edgelist)
        
        # ...Then positions
        for node, (x, y) in self.pos.iteritems():
            dx, dy = self._velocity[node]
            newtuple = (x + dx, y + dy)
            self.pos[node] = newtuple
        
        
                
    
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
    
#    def update_position(self, pos = None):
#        '''Updates node position. If no position dict is provided, generates
#        a spring layout.'''
#        if pos == None:
#            self.pos = nx.spring_layout(self)
#        else:
#            self.pos = pos  #TODO normaliser koordinater!!!
    
    def remove_nodes_from(self, nodes_to_remove):
        if self.pos:
            for node in nodes_to_remove:
                del self.pos[node]
                if node in self.infected_nodes:
                    self.infected_nodes.remove(node)
        super(TemporalGraph, self).remove_nodes_from(nodes_to_remove)

    def remove_node(self, node):
        self.remove_nodes_from(set(node))
    
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
    #Read in timeline data
    with open('temp/timeline.json', 'r') as f:
        timeline = [json.loads(line) for line in f.readlines()]
    
    G.add_timeline(timeline)
    G.update()
    G.infect_random_node()
    G.infect_random_node()
    G.infect_random_node()
    counter = 0
    while counter <= 10000 and not G.is_extinct():
        G.update()
#        G.draw(node_size = 50, show = False,
#               filename = 'temp/frames/f%03da' % counter)
        G.run_step()
        G.draw(node_size = 50, show = False,
               filename = 'temp/frames/f%03d' % counter)
        counter += 1
        print counter
        
#    G.add_timeline(timeline)
#    G.draw(node_size = 100)
#    
#    test = timeline[301]
#    
#    # Trust me on this one...
#    all_nodes = list(set(sum(test['edgelist'],[])))
#    G.add_nodes_from(all_nodes)
#    G.add_edges_from([tuple(edge) for edge in test['edgelist']])
#    #G.update_position(pos = test['positions'])
#    pos = test['positions']
#    G.draw()
#    
#    argh = []
#    for d in timeline:
#        pos = d['positions']
#        all_nodes = list(set(sum(d['edgelist'],[])))
#        argh.append(len(set(all_nodes).intersection(set(pos.keys()))))
#    
#    print dict(Counter(argh))
#
#
#
#
#
#
#
#
#
##    n_nodes = 100
##    G.add_nodes_from(range(n_nodes))
##    G.draw()
##    G.infect_random_node()
##    counter = 1
##    lim = float('inf')
##    
##    while not G.is_extinct() and counter < lim:
##        new_edges = [(np.random.randint(n_nodes), np.random.randint(n_nodes))
##                     for _ in xrange(10)]
##        G.add_edges_from(new_edges)
##        G.draw(node_size = 50, show = False, filename = 'temp/f%03da' % counter)
##        G.run_step()
##        G.draw(node_size = 50, show = False, filename = 'temp/f%03db' % counter)
##        counter += 1
##        G.run_step()
##        print counter
