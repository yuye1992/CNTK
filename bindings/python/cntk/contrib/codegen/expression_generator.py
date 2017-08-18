# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Classes and functions for expression generation from a CNTK model.
"""
from model_transforms import *
from node_visitor import *
from quantizer import *
from cntk import *
from cntk import cntk_py
from pdb import set_trace
import cntk.variables
import networkx as nx
import itertools
import functools
import json

def py2c_type(type):
    if type == np.float32:
        return 'float'
    elif type == np.float64:
        return 'double'
    else:
        raise ValueError('Unsupported type %s' % type)

class WeightsExtractor(EmptyNodeVisitor):
    '''
    Extracts weights and constants into a separate json file.
    TODO: We should take dependency on protobuf and extract 
    this values directly.
    '''
    def __init__(self, graph):
        super(EmptyNodeVisitor, self).__init__(graph)

    def dump(self, filepath):
        self.weights = {}
        self.visit(self.graph.nodes())
        json.encoder.FLOAT_REPR = lambda o: format(o, '.9f')
        with open(filepath, "w") as f:
            json.dump(self.weights, f)

    def visit_parameter(self, node):
        self.weights[node.uid] = [float(f) for f in node.as_parameter().value.flatten()]

    def visit_constant(self, node):
        self.weights[node.uid] = [float(f) for f in node.as_constant().value.flatten()]

class CppNamespaceGen:
    '''
    Helper class for generation of C++ namespace.
    '''
    def __init__(self, name):
        '''
        Constructor.
        Args:
            name(str): name of the namespace.
        '''
        self.name = name
        self.members = []


    def add_member(self, member_definition):
        '''
        Adds a member with the provided definition.
        Args:
            member_definition(str): member definition/body.
        ''' 
        self.members.append(member_definition)

    def __str__(self):
        result = []
        result.append('namespace %s' % self.name)
        result.append('{')
        result.extend(self.members)
        result.append('};')
        return '\n'.join(result)

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
        all_params = ', '.join(['const Halide::ImageParam& %s' % i for i in self.inputs])
        
        # Generating the class with setters for weights and constants. 
        evaluator = CppClassGen(class_name)
        for node in self.values:
            original_type = py2c_type(node.dtype)
            quantized_type = self.data_type(node)
            if original_type == quantized_type:
                evaluator.add_private_member('std::vector<%s> m_%s;' % (original_type, node.uid))       
                evaluator.add_public_member('const std::vector<%s> get_%s() const { return m_%s; }' % (original_type, node.uid.lower(), node.uid))
                evaluator.add_public_member('void set_%s(const std::vector<%s>&& v) { m_%s = std::move(v); };' % (node.uid.lower(), original_type, node.uid))
            else:
                evaluator.add_private_member('std::vector<%s> m_%s;' % (quantized_type, node.uid))       
                evaluator.add_private_member('%s m_step_%s;' % (original_type, node.uid))       
                evaluator.add_public_member('const std::vector<%s> get_%s() const { return m_%s; }' % (quantized_type, node.uid.lower(), node.uid))
                evaluator.add_public_member('void set_%s(const std::vector<%s>&& v) { auto r = Quantize<%s, %s>(v, %d); m_%s = r.first; m_step_%s = r.second; };' % (node.uid.lower(),
                                             original_type, original_type, quantized_type, self.graph.node[node.uid]['reserved_bits'], node.uid, node.uid))

        # Actually generating the function that will create the computation graph.
        eval_graph = 'Halide::Pipeline create_eval_graph(%s)\n {\n %s \n %s \n %s \n }\n' % (all_params, 'Halide::Var var1, var2;', self.listing, self.generate_return_value())
        evaluator.add_public_member(eval_graph)

        nspace = CppNamespaceGen('CNTK')
        nspace.add_member(str(evaluator))
        return self.generate_file_header() + str(nspace);

    def visit_parameter(self, node):
        node = node.as_parameter()
        exp = self.generate_value(node)
        self.listing += exp + '\n\n'
        self.uid_to_exp[node.uid] = '%s' % node.uid

    def visit_constant(self, node):
        node = node.as_constant()
        exp = self.generate_value(node)
        self.listing += exp + '\n\n'
        self.uid_to_exp[node.uid] = '%s' % node.uid

    def visit_input(self, node):
        input = self.graph.node[node.uid]
        if 'original' in input:
            original = input['original']
            input_name = '%s' % (node.name)
        else:
            input_name = '%s' % (node.uid)
        self.uid_to_exp[node.uid] = '%s' % input_name
        self.inputs.append(input_name)

    def visit_output(self, node):
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

    def visit_primitive_function(self, node):
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

    def visit_node(self, node):
        if isinstance(node, QuantizeNode):
            self.generate_quantization(node)
        else:
            raise ValueError('Unexpected node' % node)
        self.uid_to_exp[node.uid] = '%s' % node.uid

    def generate_quantization(self, node):
        operands = get_predecessors(self.graph, node.uid)
        if len(operands) != 1:
            raise ValueError('Operation "quantization" expects 1 operand, given %s', str(operands))
        shape = node.shape if len(node.shape) > 0 else (1,)
        shape_arg = '%d, %d' % (shape[0], shape[1]) if len(shape) == 2 else '%d' % shape[0]
        exp = 'std::vector<Halide::Func> %s; %s = Quantize<%s, %s>(%s, %s, %d)' % tuple([node.uid, node.uid, 'float' if node.dtype == np.float32 else 'double', self.data_type(node), 
                                                                                        self.uid_to_exp[operands[0]], shape_arg, self.graph.node[node.uid]['reserved_bits']] )
        self.listing += exp + ';\n'

    def generate_binary_call(self, node, op_name):
        operands = get_predecessors(self.graph, node.uid)
        if len(operands) != 2:
            raise ValueError('Operation "%s" expects 2 operands, given %s', op_name, str(operands))
        exp = 'Halide::Func %s("%s"); %s = %s(%s, %s)' % tuple([node.uid, node.uid, node.uid, op_name] + [self.uid_to_exp[o] for o in operands])
        if len(self.graph.successors(node.uid)) > 1: # Make sure we do not recalculate the same values twice.
            exp += ';\n%s.compute_root()' % node.uid
        self.listing += exp + ';\n'

    def generate_call(self, node, op_name, operands):
        str_operands = ','.join(operands)
        exp = 'Halide::Func %s("%s"); %s = %s(%s)' % tuple([node.uid, node.uid, node.uid, op_name, str_operands])
        if len(self.graph.successors(node.uid)) > 1: # Make sure we do not recalculate the same values twice.
            exp += ';\n%s.compute_root()' % node.uid
        self.listing += exp + ';\n'

    def generate_unary_call(self, node, op_name):
        operands = get_predecessors(self.graph, node.uid)
        if len(operands) != 1:
            raise ValueError('Operation "%s" expects 1 operand, given %s', op_name, str(operands))
        exp = 'Halide::Func %s("%s"); %s = %s(%s)' % tuple([node.uid, node.uid, node.uid, op_name, self.uid_to_exp[operands[0]]])
        self.listing += exp + ';\n'

    def generate_times(self, node):
        operands = get_predecessors(self.graph, node.uid)
        node_attrs = self.graph.node[node.uid]
        if len(operands) != 2:
            raise ValueError('Expecting 2 operands')
        vector = self.graph.node[operands[0]]['data']
        if len(vector.shape) != 0 and len(vector.shape) != 1:
            set_trace()
            raise ValueError("Times is currently supported only for 1D * 2D, given %s" % str(vector.shape))

        matrix = self.graph.node[operands[1]]['data']
        shape = matrix.shape if len(matrix.shape) > 0 else (1,)        

        op_name = 'VectorByMatrixTimes'
        if 'quantized' in self.graph.node[node.uid]:
            op_name += 'Quantized'
        self.generate_call(node, op_name, [self.uid_to_exp[o] for o in operands] + [str(shape[0]), str(shape[1])])

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
            exp = 'Halide::Func %s("%s"); %s = Slice<%d, %d>(%s)' % (node.uid, node.uid, node.uid, begin, end, self.uid_to_exp[operand.uid])
        else:
            raise ValueError('Slice is not supported on node of shape %s' % str(node.shape)) 
        self.listing += exp + ';\n'

    def generate_file_header(self):
        header  = '#pragma once\n'
        header += '#include "HalideDNNLib.h"\n\n'
        return header;

    def generate_return_value(self):
        return 'return Halide::Pipeline({ %s });' % ', '.join(self.outputs)

    def data_type(self, node):
        node_attrs = self.graph.node[node.uid]
        if 'quantized' in node_attrs:
            if node_attrs['total_bits'] == 16:
                return 'short'
            elif node_attrs['total_bits'] == 8:
                return 'char'
            else:
                raise ValueError('Unsupported type')
        else:
            return 'float' if node.dtype == np.float32 else 'double'

    def total_num_elements(self, shape):
        return shape[0] if len(shape) == 1 else 1 if len(shape) == 0 else functools.reduce(lambda x, y: x*y, shape)

    def generate_value(self, node):
        type = self.data_type(node)
        if len(node.shape) == 2:
            expression = 'auto b_%s = Halide::Buffer<%s>(m_%s.data(), %s, %s, "%s");\n' % (node.uid, type, node.uid, node.shape[1], node.shape[0], node.uid)
        elif len(node.shape) == 1:
            expression = 'auto b_%s = Halide::Buffer<%s>(m_%s.data(), %s, "%s");\n' % (node.uid, type, node.uid, node.shape[0], node.uid)
        elif len(node.shape) == 0: # Scalar represent as array
            expression = 'auto b_%s = Halide::Buffer<%s>(m_%s.data(), %s, "%s");\n' % (node.uid, type, node.uid, 1, node.uid)
        else:
            set_trace()
            raise ValueError('Unexpected shape encountered, only 1 and 2D are currently supported %s' % node)
        
        if 'quantized' not in self.graph.node[node.uid]:
            expression += 'Halide::Func %s("%s"); %s(%s) = b_%s(%s);' % (node.uid, node.uid, node.uid, self.index_vars(node), node.uid, self.index_vars(node))
        else:
            expression += 'Halide::Func f_%s("f_%s"); f_%s(%s) = b_%s(%s);\n' % (node.uid, node.uid, node.uid, self.index_vars(node), node.uid, self.index_vars(node))
            expression += 'Halide::Func f_step_%s("f_step_%s"); f_step_%s() = m_step_%s;\n' % (node.uid, node.uid, node.uid, node.uid)
            expression += 'std::vector<Halide::Func> %s { f_%s, f_step_%s };\n' % (node.uid, node.uid, node.uid)
        self.values.append(node)
        return expression
