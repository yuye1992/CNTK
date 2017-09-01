# ==========================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Different quantization algorithms.
"""
from model_transforms import *
from node_visitor import EmptyNodeVisitor
from pdb import set_trace
import numpy as np
import networkx as nx

class QuantizeNode(Node):
    def __init__(self, graph, name, shape, dtype):
        super(QuantizeNode, self).__init__(graph, name, shape)
        self.name = name
        self.op_name = 'Quantize'
        self.shape = shape
        self.dtype = dtype

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
        if node.op_name == 'Times':
            self.quantize_times(node)

    def quantize_times(self, node):
        node.quantize = True
        for p in node.predecessors:
            node_name = p + '_' + node.id + '_quantize'
            self.graph.add_node(QuantizeNode(name=node_name, shape=p.shape, dtype=p.dtype))
            self.graph.add_edge(p, node_name, order=0)
            self.graph.add_edge(node_name, node.uid, order=self.graph.get_edge_data(p, node)['order'])
            self.graph.remove_edge(p, node)

    def visit_parameter(self, node):
        successors = self.graph.successors(node)
        all_quantized = (sum([0 if isinstance(s, QuantizeNode) else 1 for s in successors]) == 0)
        if not all_quantized:
            return

        # Mark parameter to be quantized.
        node.quantize = True

        # Remove all quantization nodes after it.
        for s in successors:
            for grand in self.graph.successors(s):
                self.graph.add_edge(node, grand, order=self.graph.get_edge_data(s, grand)['order'])
            self.graph.remove_node(s)
