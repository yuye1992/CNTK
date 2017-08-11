# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Classes and functions for expression generation from a CNTK model.
"""
from .model_transforms import *
from cntk import *
from cntk import cntk_py
from pdb import set_trace
import cntk.variables
import networkx as nx
import itertools
import functools

class NodeVisitor:
    '''
    Visitor for the model nodes.
    '''
    def __init__(self, graph):
        '''
        Constructor.
        Args:
            graph: nx graph of the model     
        '''
        super(NodeVisitor, self).__init__()
        self.graph = graph

    def visit(self, nodes):
        '''
        Visits the nodes in order.
        Args:
            nodes(list): uids of nodes for evaluation. Nodes are evaluated
              in the given list order.
        '''
        from cntk import cntk_py

        for n in nodes:
            node = self.graph.node[n]['data']
            if isinstance(node, cntk_py.Function):
                if not node.is_primitive:
                    raise ValueError('Unexpected non primitive function %s' % node)
                self.visit_primitive_function(node)
            elif node.is_parameter:
                self.visit_parameter(node)
            elif node.is_constant:
                self.visit_constant(node)
            elif node.is_input:
                self.visit_input(node)
            elif node.is_output:
                self.visit_output(node)
            else:
                raise ValueError("Unexpected node type %s" % node)

    def visit_parameter(self, node):
        raise NotImplemented()

    def visit_constant(self, node):
        raise NotImplemented()

    def visit_input(self, node):
        raise NotImplemented()

    def visit_output(self, node):
        raise NotImplemented()

    def visit_primitive_function(self, node):
        raise NotImplemented()

class EmptyNodeVisitor(NodeVisitor):
    '''
    Empty node visitor for the model nodes.
    '''
    def __init__(self, graph):
        super(EmptyNodeVisitor, self).__init__(graph)

    def visit_parameter(self, node):
        pass

    def visit_constant(self, node):
        pass

    def visit_input(self, node):
        pass

    def visit_output(self, node):
        pass

    def visit_primitive_function(self, node):
        pass


class WeightsExtractor(EmptyNodeVisitor):
    '''
    Extracts weights and constants into a separate file.
    TODO: We should take dependency on protobuf and extract 
    this values directly.
    '''
    def __init__(self, graph):
        super(EmptyNodeVisitor, self).__init__(graph)
        self.weights = {}

    def dump(self, filepath):
        self.visit(self.nodes)
        json.encoder.FLOAT_REPR = lambda o: format(o, '.9f')
        with open(filepath, "w") as f:
            json.dump(weights, f)

    def visit_parameter(self, node):
        weights[node.uid] = [float(f) for f in node.value.flatten()]

    def visit_constant(self, node):
        weights[node.uid] = [float(f) for f in node.value.flatten()]

class CppClassGen:
    '''
    Helper class for generation of C++ class.
    '''
    def __init__(self, name):
        '''
        Constructor.
        Args:
            name(str): name of the class.
        '''
        self.public = []
        self.private = []
        self.name = name

    def add_private_member(self, member_definition):
        '''
        Adds a private member with the provided definition.
        Args:
            member_definition(str): member definition/body.
        ''' 
        self.private.append(member_definition)

    def add_public_member(self, member_definition):
        '''
        Adds a public member with the provided definition.
        Args:
            member_definition(str): member definition/body.
        ''' 
        self.public.append(member_definition)

    def __str__(self):
        result = []
        result.append('class %s final' % self.name)
        result.append('{')
        if len(self.public) > 0:
            result.append('public:')
            for m in self.public:
                result.append(str(m))
        if len(self.private) > 0:
            result.append('private:')
            for m in self.private:
                result.append(str(m))
        result.append('};')
        return '\n'.join(result)

class HalideExpressionGenerator(NodeVisitor):
    '''
    Generator of halide graph from the NX model graph.
    '''
    def __init__(self, graph):
        super(HalideExpressionGenerator, self).__init__(graph)
        self.uid_to_exp = {}
        self.listing = ''
        self.inputs = []
        self.outputs = []
        self.values = []

    def generate(self, nodes, class_name='evaluator'):
        self.visit(nodes)
        all_params = ', '.join(self.inputs)
        
        # Generating the class with setters for weights and constants. 
        evaluator = CppClassGen(class_name)
        for node in self.values:
            evaluator.add_private_member('std::vector<%s> m_%s;' % (self.data_type(node), node.uid))       
            evaluator.add_public_member('const std::vector<%s> get_%s() const { return m_%s; }' % (self.data_type(node), node.uid.lower(), node.uid))
            evaluator.add_public_member('void set_%s(const std::vector<%s>&& v) { m_%s = std::move(v); };' % (node.uid.lower(), self.data_type(node), node.uid))

        # Actually generating the function that will create the computation graph.
        eval_graph = 'Pipeline create_eval_graph(%s)\n {\n %s \n %s \n %s \n }\n' % 
                      (all_params, 'Var var1, var2;', self.listing, self.generate_return_value())
        evaluator.add_public_member(eval_graph)

        return self.generate_file_header() + str(evaluator);

    def generate_file_header(self):
        header  = '#pragma once\n'
        header += '#include "HalideDNNLib.h"\n\n'
        return header;

    def generate_return_value(self):
        return 'return Pipeline({ %s });' % ', '.join(self.outputs)

    def data_type(self, node):
        return 'float' if node.dtype == np.float32 else 'double'

    def total_num_elements(self, shape):
        return shape[0] if len(shape) == 1 else 1 if len(shape) == 0 else functools.reduce(lambda x, y: x*y, shape)

    def generate_value(self, node):
        if node.dtype == np.float32:
            data_type = 'float'
        else:
            assert node.dtype == np.float64
            data_type = 'double'
        if len(node.shape) == 2:
            expression = 'auto b_%s = Halide::Buffer<%s>(m_%s.data(), %s, %s, "%s");\n' % (node.uid, data_type, node.uid, node.shape[0], node.shape[1], node.uid)
        elif len(node.shape) == 1:
            expression = 'auto b_%s = Halide::Buffer<%s>(m_%s.data(), %s, "%s");\n' % (node.uid, data_type, node.uid, node.shape[0], node.uid)
        elif len(node.shape) == 0: # Scalar represent as array
            expression = 'auto b_%s = Halide::Buffer<%s>(m_%s.data(), %s, "%s");\n' % (node.uid, data_type, node.uid, 1, node.uid)
        else:
            set_trace()
            raise ValueError('Unexpected shape encountered, only 1 and 2D are currently supported %s' % node)

        expression += 'Func %s("%s"); %s(%s) = b_%s(%s);' % (node.uid, node.uid, node.uid, self.index_vars(node), node.uid, self.index_vars(node))
        self.values.append(node)
        return expression

    def generate_parameter(self, node):
        node = node.as_parameter()
        exp = self.generate_value(node)
        self.listing += exp + '\n\n'
        self.uid_to_exp[node.uid] = '%s' % node.uid

    def generate_constant(self, node):
        node = node.as_constant()
        exp = self.generate_value(node)
        self.listing += exp + '\n\n'
        self.uid_to_exp[node.uid] = '%s' % node.uid

    def generate_input(self, node):
        input = self.graph.node[node.uid]
        if 'original' in input:
            original = input['original']
            input_name = '%s' % (node.name)
        else:
            input_name = '%s' % (node.uid)
        self.uid_to_exp[node.uid] = '%s' % input_name
        self.inputs.append(input_name)

    def generate_output(self, node):
        output = self.graph.node[node.uid]
        operands = get_predecessors(self.graph, node.uid)
        if 'original' in output:
            original = output['original']
            if len(operands) != 2:
                raise ValueError('Past Value value is expected to have 2 operands, given %s' % str(operands))
        else:
            if len(operands) != 1:
                raise ValueError('Output value is expected to have a single operand, given %s' % str(operands))
        self.outputs.append(operands[0])

    def index_vars(self, node):
        if len(node.shape) == 1 or len(node.shape) == 0:
            return 'var1'
        elif len(node.shape) == 2:
            return 'var1, var2'
        else:
            set_trace()
            raise ValueError("Shape is not supported %s" % str(node.shape))

    def generate_primitive_function(self, node):
        op_name = node.op_name
        if op_name == 'Times':
            self.generate_times(node)
        elif op_name == 'Plus':
            self.generate_plus(node)
        elif op_name == 'Slice':
            self.generate_slice(node)
        elif op_name == 'StableSigmoid':
            self.generate_stable_sigmoid(node)
        elif op_name == 'Tanh':
            self.generate_tanh(node)
        elif op_name == 'ElementTimes':
            self.generate_element_times(node)
        else:
            set_trace()
            raise ValueError('Not implemented function %s' % node.op_name)
        self.uid_to_exp[node.uid] = '%s' % node.uid

    def generate_binary_call(self, node, op_name):
        operands = get_predecessors(self.graph, node.uid)
        if len(operands) != 2:
            raise ValueError('Operation "%s" expects 2 operands, given %s', op_name, str(operands))
        exp = 'Func %s("%s"); %s = %s(%s, %s)' % tuple([node.uid, node.uid, node.uid, op_name] + [self.uid_to_exp[o] for o in operands])
        if len(self.graph.successors(node.uid)) > 1: # Make sure we do not recalculate the same values twice.
            exp += ';\n%s.compute_root()' % node.uid
        self.listing += exp + ';\n'

    def generate_unary_call(self, node, op_name):
        operands = get_predecessors(self.graph, node.uid)
        if len(operands) != 1:
            raise ValueError('Operation "%s" expects 1 operand, given %s', op_name, str(operands))
        exp = 'Func %s("%s"); %s = %s(%s)' % tuple([node.uid, node.uid, node.uid, op_name, self.uid_to_exp[operands[0]]])
        self.listing += exp + ';\n'

    def generate_times(self, node):
        operands = get_predecessors(self.graph, node.uid)
        if len(operands) != 2:
            raise ValueError('Expecting 2 operands')
        operand = self.graph.node[operands[0]]['data']
        if len(operand.shape) != 0 and len(operand.shape) != 1:
            set_trace()
            raise ValueError("Times is currently supported only for 1D * 2D, given %s" % str(operand.shape))
        shape = operand.shape if len(operand.shape) == 1 else (1,)        
        self.generate_binary_call(node, 'VectorByMatrixTimes<%s, %s>' % (shape[0], node.shape[0 if len(node.shape) == 1 else 1]))

    def generate_element_times(self, node):
        self.generate_binary_call(node, 'ElementTimes')

    def generate_plus(self, node):
        self.generate_binary_call(node, 'Plus')

    def generate_stable_sigmoid(self, node):
        self.generate_unary_call(node, 'Sigmoid<%s>' % self.data_type(node))

    def generate_tanh(self, node):
        self.generate_unary_call(node, 'Tanh')

    def generate_slice(self, node):
        operand = get_predecessors(self.graph, node.uid)
        if len(operand) != 1:
            raise ValueError('Operation "slice" expects 2 operands')
        operand = self.graph.node[operand[0]]['data']
        if len(operand.shape) == 1:
            begin = node.attributes['beginIndex']
            end = node.attributes['endIndex']
            stride = node.attributes['sliceStrides']
            if stride != 1:
                raise ValueError('Unexpected stride "%s", only stride of 1 is currently supported' % str(stride))
            exp = 'Func %s("%s"); %s = Slice<%d, %d>(%s)' % (node.uid, node.uid, node.uid, begin, end, self.uid_to_exp[operand.uid])
        else:
            raise ValueError('Slice is not supported on node of shape %s' % str(node.shape)) 
        self.listing += exp + ';\n'
