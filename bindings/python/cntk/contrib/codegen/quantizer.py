# ==========================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Different quantization algorithms.
"""
from node_visitor import EmptyNodeVisitor
from model_transforms import get_predecessors
from pdb import set_trace
import numpy as np
import networkx as nx

class QuantizeNode:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.uid = name
        self.op_name = 'quantize'
        self.shape = shape
        self.dtype = dtype

    @property
    def is_input(self):
        return False;

    @property
    def is_placeholder(self):
        return False

    @property
    def is_parameter(self):
        return False

    @property
    def is_constant(self):
        return False

    @property
    def is_output(self):
        return False

    @property
    def dynamic_axes(self):
        return []

class OperationQuantizer(EmptyNodeVisitor):
    '''
    Quantizes the nodes that support quantization.
    '''
    def __init__(self, graph, quantization_method, reserved_bits, total_bits):
        super(EmptyNodeVisitor, self).__init__(graph)
        self.method = quantization_method       
        if self.method != 'symmetric':
            raise ValueError('Currently only symmetric quantization is supported')

        self.reserved_bits = int(reserved_bits)
        self.total_bits = int(total_bits)

        if self.reserved_bits >= self.total_bits:
            raise ValueError('Value of reserved_bits cannot exceed the total_bits value')

    def quantize(self, nodes):
        self.visit(nodes)

    def visit_primitive_function(self, node):
        op_name = node.op_name
        if op_name == 'Times':
            self.quantize_times(node)

    def quantize_times(self, node):
        self.graph.node[node.uid]['quantized'] = True
        self.graph.node[node.uid]['total_bits'] = self.total_bits
        self.graph.node[node.uid]['reserved_bits'] = self.reserved_bits
        predecessors = get_predecessors(self.graph, node.uid)
        for p in predecessors:
            node_name = p + '_' + node.uid + '_quantize'
            self.graph.add_edge(p, node_name, order=0)
            self.graph.add_edge(node_name, node.uid, order=self.graph.get_edge_data(p, node.uid)['order'])
            self.graph.node[node_name]['data'] = QuantizeNode(name=node_name, shape=self.graph.node[p]['data'].shape, dtype=node.dtype)
            self.graph.node[node_name]['quantized'] = True
            self.graph.node[node_name]['reserved_bits'] = self.reserved_bits
            self.graph.node[node_name]['total_bits'] = self.total_bits
            self.graph.remove_edge(p, node.uid)

    def visit_parameter(self, node):
        successors = self.graph.successors(node.uid)
        all_quantized = (sum([0 if isinstance(self.graph.node[s]['data'], QuantizeNode) else 1 for s in successors]) == 0)
        if not all_quantized:
            return

        # Mark parameter to be quantized.
        self.graph.node[node.uid]['quantized'] = True 
        self.graph.node[node.uid]['reserved_bits'] = self.reserved_bits
        self.graph.node[node.uid]['total_bits'] = self.total_bits

        # Remove all quantization nodes after it.
        for s in successors:
            for grand in self.graph.successors(s):
                self.graph.add_edge(node.uid, grand, order=self.graph.get_edge_data(s, grand)['order'])
            self.graph.remove_node(s)
