# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Classes and functions for building a NetworkX graph of a CNTK model.
"""

from cntk import *
from cntk import cntk_py
from pdb import set_trace
import cntk.variables
import networkx as nx

def get_predecessors(graph, node):
    '''
    Utility function to return all predecessors of a node from the graph honouring the order.
    The order is taken from the 'order' attribut of the corresponding (predecessor, node) edge.
    Args:
        graph(list): nx model graph
        node(str): uid of the model node for which to find the predecessors

    Returns:
        list of predecessor nodes in the order according to the 'order' attribute of the edges.
    '''
    predecessors = graph.predecessors(node)     
    ordered = [(p, graph.get_edge_data(p, node)['order']) for p in predecessors]
    ordered = sorted(ordered, key=lambda t: t[1])
    return [o[0] for o in ordered]

class ModelToGraphConverter:
    '''
    Converts a CNTK model to a NX graph. Eliminates block functions.
    Each node in the graph contains the uid of the corresponding entity of the model.
    The original model node is stored in the attached 'data' attribute.
    Because the order of edges is not defined for NX graphs, each edge has an additional 
    numeric attribute 'order' that corresponds to the order of parameters.
    TODO: Currently does not handle nested blocks
    '''
    def __init__(self):
        super(ModelToGraphConverter, self).__init__()

    def convert(self, model):
        '''
        Converts CNTK model to the NX graph.
        Args:
            model: CNTK model

        Returns:
            NX graph that corresponds to the model.
        '''
        from cntk import cntk_py
        outputs = []
        if isinstance(model, cntk_py.Function):
            if model.is_composite:
                model = model.root_function
            outputs.extend(model.outputs)
        elif isinstance(model, cntk_py.Variable):
            outputs = [model]
        else:
            raise ValueError('Model is expected to be an output variable or a function')

        g = nx.OrderedDiGraph()
        visited = {}
        for output in model.outputs:
            self._convert(g, output, None, 0, set(), {})
        return g

    def _convert(self, g, node, child, order, visited, placeholder_mapping):
        from cntk import cntk_py
        is_function = isinstance(node, cntk_py.Function)

        # First thing - add an edge between the child and the node
        # skipping blocks if needed
        # BUGBUG: Two nested blocks?
        if child is not None:
            if not g.has_node(child.uid):
                g.add_node(child.uid, data=child)
            cur = dict(node.block_outputs_mapping)[child] if is_function and node.is_block else node
            if not g.has_node(cur.uid):
                g.add_node(cur.uid, data=cur)
            g.add_edge(cur.uid, child.uid, order=order)

        if node.uid in visited:
            return
        visited.add(node.uid)

        if is_function:
            if node.is_block:
                placeholder_mapping.update(node.block_arguments_mapping)
                outputs_mapping = dict(node.block_outputs_mapping)
                inner_output_variable = outputs_mapping[child]
                self._convert(g, inner_output_variable, child, order, visited, placeholder_mapping)
            elif node.is_primitive:
                for order, i in enumerate(node.inputs):
                    i = placeholder_mapping[i] if i.is_placeholder else i
                    self._convert(g, i, node, order, visited, placeholder_mapping)
            else:
                set_trace()
                raise ValueError("Unexpected function node type %s" % node)

        elif node.is_parameter or node.is_constant or node.is_input:
            pass
        elif node.is_output:
            self._convert(g, node.owner, node, order, visited, placeholder_mapping)
        elif node.is_placeholder:
            actual_node = placeholder_mapping[node]
            self._convert(g, actual_node, order, visited, placeholder_mapping)
        else:
            set_trace()
            raise ValueError("Unexpected node type %s" % node)


def remove_intermediate_output_nodes(graph):
    '''
    Utility function to remove intermediate output variables from the graph.
    Only actual outputs of the graph are preserved.
    Args:
        graph: nx model graph
    '''
    # Remove all output variables in the graph
    # except for the actual end outputs (that have no children).
    removed = True
    while removed:
        removed = False
        for n in graph.nodes():
            node = graph.node[n]['data']
            if not (isinstance(node, cntk.variables.Variable) and node.is_output):
                continue
     
            successors = graph.successors(n)
            if len(successors) == 0: # No successors - actual output
                continue
     
            predecessors = get_predecessors(graph, n)
            if len(predecessors) != 1:
                raise ValueError("Unexpected output node with no ancestors")

            p = predecessors[0] 
            for s in successors:
                graph.add_edge(p, s, data=graph.node[n]['data'], 
                               label=graph.node[n]['data'].uid, order=graph.get_edge_data(n, s)['order'])
            graph.remove_node(n)
            removed = True

def split_past_values(graph):
    '''
    Splits each past value into input and output past value.
    TODO: Initial non zero state is not yet handled correctly. 
    Args:
        graph: nx model graph
    '''
    for n in graph.nodes():
        node = graph.node[n]['data']
        if not isinstance(node, cntk_py.Function):
            continue
        if node.op_name != 'PastValue':
            continue

        external_output = cntk.output_variable(dynamic_axes=node.dynamic_axes, shape=node.shape, dtype=node.dtype, name=node.uid)
        external_input = cntk.input_variable(dynamic_axes=node.dynamic_axes, shape=node.shape, dtype=node.dtype, name=node.uid)

        graph.add_node(external_input.uid, data=external_input, original=node)
        graph.add_node(external_output.uid, data=external_output, original=node)

        for successor in graph.successors(n):
            graph.add_edge(external_input.uid, successor, order = graph.get_edge_data(n, successor)['order'])

        for predecessor in get_predecessors(graph, n):
            graph.add_edge(predecessor, external_output.uid, order = graph.get_edge_data(predecessor, n)['order'])

        graph.remove_node(n)
