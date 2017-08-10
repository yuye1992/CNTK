from cntk import *
from cntk import cntk_py
from cntk.logging import *
import cntk.variables
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pdb import set_trace
import itertools
import functools

model_path = r'c:\repo\halide_playground\my_super.model'
#model_path = r'c:\repo\halide_playground\test_simple_model'

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

def get_predecessors(graph, node):
    predecessors = g.predecessors(node)     
    ordered = [(p, graph.get_edge_data(p, node)['order']) for p in predecessors]
    ordered = sorted(ordered, key=lambda t: t[1])
    return [o[0] for o in ordered]

# Utility function for NX Graph visualization
def nx_plot(g, filename):
    if filename:
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in ('.svg', '.pdf', '.png', '.dot'):
            raise ValueError('only file extensions ".svg", ".pdf", ".png", and ".dot" are supported')
    else:
        suffix = None

    if filename:
        try:
            import pydot_ng as pydot
        except ImportError:
            raise ImportError("Unable to import pydot_ng, which is required to output SVG, PDF, PNG, and DOT format.")

        # initialize a dot object to store vertices and edges
        dot_object = pydot.Dot(graph_name="network_graph", rankdir='TB')
        dot_object.set_node_defaults(shape='rectangle', fixedsize='false',
                                     style='filled',
                                     fillcolor='lightgray',
                                     height=.85, width=.85, fontsize=12)
        dot_object.set_edge_defaults(fontsize=10)

    primitive_op_map = {
        'Plus': '+',
        'Minus': '-',
        'ElementTimes': '*',
        'Times': '@',
    }
    dot_nodes = {}  # [uid] -> dot node

    def node_desc(node):
        return '<' + node.uid + '>'

    def shape_desc(node):
        dyn_axes = node.dynamic_axes
        dyn = '[#' + ',*' * (len(dyn_axes) - 1) + ']' if len(dyn_axes) > 0 else ''
        return dyn + str(node.shape)

    # add current Function node
    def create_node(node):
        if node.uid in dot_nodes: # dot node already exists
            raise ValueError('Node is already created')

        if node.is_primitive and not node.is_block and len(node.outputs) == 1 and node.output.name == node.name:     # skip the node name if redundant
            op_name = primitive_op_map.get(node.op_name, node.op_name)
            render_as_primitive = len(op_name) <= 4
            size = 0.4 if render_as_primitive else 0.6
            cur_node = pydot.Node(node.uid, label='"' + op_name + node_desc(node) + '"',
                                  shape='ellipse'  if render_as_primitive else 'box',
                                  fixedsize='true' if render_as_primitive else 'false', height=size, width=size,
                                  fontsize=20  if render_as_primitive and len(op_name) == 1 else 12 ,
                                  penwidth=4 if node.op_name != 'Pass' and node.op_name != 'ParameterOrder' else 1)
        else:
            f_name = '\n' + node.name + '()' if node.name else ''
            cur_node = pydot.Node(node.uid, label='"' + node.op_name + f_name + node_desc(node) + '"',
                                  fixedsize='true', height=1, width=1.3,
                                  penwidth=4 if node.op_name != 'Pass' and node.op_name != 'ParameterOrder' else 1)
        dot_object.add_node(cur_node)
        dot_nodes[node.uid] = cur_node
        return cur_node

    # Add all nodes
    for n in g.nodes():
        node = g.node[n]['data']
        from cntk import cntk_py
        if isinstance(node, cntk_py.Function):
            # add current node
            cur_node = create_node(node)
            dot_object.add_node(cur_node)
            continue
        elif node.is_input:
            shape = 'invhouse'
            color = 'yellow'
        elif node.is_placeholder:
            shape = 'invhouse'
            color = 'grey'
        elif node.is_parameter:
            shape = 'diamond'
            color = 'green'
        elif node.is_constant:
            shape = 'rectangle'
            color = 'lightblue'
        else: # is_output
            shape = 'invhouse'
            color = 'grey'

        name = 'Parameter' if node.is_parameter else 'Constant' if node.is_constant else 'Input' if node.is_input else 'Placeholder' if node.is_placeholder else 'Output'
        if node.name:
            if name == 'Parameter':  # don't say 'Parameter' for named parameters, it's already indicated by being a box
                name = node.name
            else:
                name = name + '\n' + node.name
        name += '\n' + shape_desc(node) + '\n' + node_desc(node)
        if node.is_input or node.is_placeholder: # graph inputs are eggs (since dot has no oval)
            cur_node = pydot.Node(node.uid, shape='egg', label=name, fixedsize='true', height=1, width=1.3, penwidth=4) # wish it had an oval
        elif not node.name and node.is_constant and (node.shape == () or node.shape == (1,)): # unnamed scalar constants are just shown as values
            cur_node = pydot.Node(node.uid, shape='box', label=str(node.as_constant().value), color='white', fillcolor='white', height=0.3, width=0.4)
        else:                                      # parameters and constants are boxes
            cur_node = pydot.Node(node.uid, shape='box', label=name, height=0.6, width=1)

        dot_object.add_node(cur_node)
        dot_nodes[node.uid] = cur_node

    # Add edges
    for n in g.nodes():
        node = g.node[n]['data']
        successors = g.successors(node.uid)
        for successor in successors:
            label = node.name if node.name else node.uid # the Output variables have no name if the function has none
            label += '\n' + shape_desc(node) + '\n' + node_desc(node)

            dot_object.add_edge(pydot.Edge(dot_nodes[node.uid], dot_nodes[successor], label=label))

    if filename:
        if suffix == '.svg':
            dot_object.write_svg(filename, prog='dot')
        elif suffix == '.pdf':
            dot_object.write_pdf(filename, prog='dot')
        elif suffix == '.png':
            dot_object.write_png(filename, prog='dot')
        else:
            dot_object.write_raw(filename)

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

class ModelToGraphConverter:
    def __init__(self):
        super(ModelToGraphConverter, self).__init__()

    def convert(self, model):
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


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# Utility functions for graph transformations

def remove_output_nodes(g):
    # Remove all output variables in the graph
    # except for the actual end outputs (that have no children).
    removed = True
    while removed:
        removed = False
        for n in g.nodes():
            node = g.node[n]['data']
            if not (isinstance(node, cntk.variables.Variable) and node.is_output):
                continue
     
            successors = g.successors(n)
            if len(successors) == 0:
                continue
     
            predecessors = get_predecessors(g, n)      
            if len(predecessors) != 1:
                raise ValueError("Unexpected output node with no ancestors")

            p = predecessors[0] 
            for s in successors:
                g.add_edge(p, s, data = g.node[n]['data'], label = g.node[n]['data'].uid, order=g.get_edge_data(n, s)['order'])
     
            g.remove_node(n)
            removed = True

def split_past_values(g):
    for n in g.nodes():
        node = g.node[n]['data']
        if not isinstance(node, cntk_py.Function):
            continue
        if node.op_name != 'PastValue':
            continue

        external_output = cntk.output_variable(dynamic_axes=node.dynamic_axes, shape=node.shape, dtype=node.dtype, name=node.uid + '_external_output')
        external_input = cntk.input_variable(dynamic_axes=node.dynamic_axes, shape=node.shape, dtype=node.dtype, name=node.uid + '_external_input')

        g.add_node(external_input.uid, data=external_input, original=node)
        g.add_node(external_output.uid, data=external_output, original=node)

        for successor in g.successors(n):
            g.add_edge(external_input.uid, successor, order = g.get_edge_data(n, successor)['order'])

        for predecessor in get_predecessors(g, n):
            g.add_edge(predecessor, external_output.uid, order = g.get_edge_data(predecessor, n)['order'])

        g.remove_node(n)

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

class ExpressionGenerator:
    def __init__(self, graph):
        super(ExpressionGenerator, self).__init__()
        self.graph = graph

    def generate(self, nodes):
        from cntk import cntk_py

        for n in nodes:
            node = self.graph.node[n]['data']
            if isinstance(node, cntk_py.Function):
                if not node.is_primitive:
                    raise ValueError('Unexpected non primitive function %s' % node)
                self.generate_primitive_function(node)
            elif node.is_parameter:
                self.generate_parameter(node)
            elif node.is_constant:
                self.generate_constant(node)
            elif node.is_input:
                self.generate_input(node)
            elif node.is_output:
                self.generate_output(node)
            else:
                raise ValueError("Unexpected node type %s" % node)

    def generate_parameter(self, node):
        raise NotImplemented()

    def generate_constant(self, node):
        raise NotImplemented()

    def generate_input(self, node):
        raise NotImplemented()

    def generate_output(self, node):
        raise NotImplemented()

    def generate_primitive_function(self, node):
        raise NotImplemented()


class HalideExpressionGenerator(ExpressionGenerator):
    def __init__(self, g):
        super(HalideExpressionGenerator, self).__init__(g)
        self.uid_to_exp = {}
        self.graph = g
        self.listing = ''
        self.outputs = []

    def generate_eval_graph(self, nodes):
        self.generate(nodes)
        inputs = []
        outputs = []
        for n in self.graph.nodes():
            n = self.graph.node[n]
            node = n['data']
            if not isinstance(node, cntk_py.Variable):
                continue

            if node.is_input:
               inputs += [node.name if 'original' in n else node.uid]
            elif node.is_output:
               outputs += [node.name if 'original' in n else node.uid]
        inputs = ['const ImageParam& i_' + i for i in inputs]

        all_params = ', '.join(inputs)
        headers = '#pragma once\n'
        headers += '#include "TestCommon.h"\n'
        function = 'Pipeline eval_graph(%s)\n {\n %s \n %s \n %s \n }\n' % (all_params, 'Var var1, var2;', self.listing, self.generate_return_value())
        return headers + function

    def generate_return_value(self):
        return 'return Pipeline({ %s });' % ', '.join(self.outputs)

    def data_type(self, node):
        return 'float' if node.dtype == np.float32 else 'double'

    def total_num_elements(self, shape):
        if len(shape) == 0:
            return 1
        if len(shape) == 1:
            return shape[0]
        return shape[0] if len(shape) == 1 else 1 if len(shape) == 0 else functools.reduce(lambda x, y: x*y, shape)

    def generate_value(self, node):
        if node.dtype == np.float32:
            data_type = 'float'
        else:
            assert node.dtype == np.float64
            data_type = 'double'
        expression = 'const %s c_%s[%d]= {' % (data_type, node.uid, self.total_num_elements(node.shape))        
        for i, value in enumerate(node.value.flatten()):
            if i % 100 == 0:
                expression += '\n'
            expression += ('%ff' % value) + ', '
            # BUGBUG - c++ compiler cannot handle such huge arrays
            if i > 1000:
                break
        expression += '};\n'
        if len(node.shape) == 2:
            expression += 'auto b_%s = Halide::Buffer<%s>((%s*)c_%s, %s, %s, "%s");\n' % (node.uid, data_type, data_type, node.uid, node.shape[0], node.shape[1], node.uid)
        elif len(node.shape) == 1:
            expression += 'auto b_%s = Halide::Buffer<%s>((%s*)c_%s, %s, "%s");\n' % (node.uid, data_type, data_type, node.uid, node.shape[0], node.uid)
        elif len(node.shape) == 0: # Scalar represent as array
            expression += 'auto b_%s = Halide::Buffer<%s>((%s*)c_%s, %s, "%s");\n' % (node.uid, data_type, data_type, node.uid, 1, node.uid)
        else:
            set_trace()
            raise ValueError('Unexpected shape encountered, only 1 and 2D are currently supported %s' % node)

        expression += 'Func %s("%s"); %s(%s) = b_%s(%s);' % (node.uid, node.uid, node.uid, self.index_vars(node), node.uid, self.index_vars(node))
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

        exp = 'Func %s("%s"); %s(%s) = i_%s(%s);' % (input_name, input_name, input_name, self.index_vars(node), input_name, self.index_vars(node))
        self.listing += exp + '\n\n'
        self.uid_to_exp[node.uid] = '%s' % input_name

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

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################



model = Function.load(model_path)

c = ModelToGraphConverter()
g = c.convert(model)

#if not nx.is_connected(g.to_undirected()):
#    raise ValueError('Unsupported type of graph: only fully connected graphs are supported.')

nx_plot(g, 'full_graph_with_outputs.pdf')

remove_output_nodes(g)
nx_plot(g, 'graph_without_outputs.pdf')

split_past_values(g)
nx_plot(g, 'DAG.pdf')

if not nx.is_directed_acyclic_graph(g):
    raise ValueError('Unsupported type of graph: please make sure there are no several past values in a single loop')

nodes_sorted_for_generation = nx.topological_sort(g)

for node in nodes_sorted_for_generation:
    print('Node name %s, uid %s' % (g.node[node]['data'].name, node))

generator = HalideExpressionGenerator(g)
generated = generator.generate_eval_graph(nodes_sorted_for_generation)

with open(r'..\amitlstm\LSTM\execution_graph.h', 'w') as f:
    f.write(generated)

#for node in g.nodes():
#    print('Node id %s' % node.uid)
#    for i in g.adj[node]:
#        print('       Connected to %s' % i.uid)



#outputs = { o for o in model.outputs }
#vars_in_eval_order = model.get_evaluation_order(outputs)


#for v in vars_in_eval_order:
#    set_trace()
#    if v.is_parameter:
#        print('Need to evaluate Parameter %s' % v.uid)
#    elif v.is_constant:
#        print('Need to evaluate Constant %s' % v.uid)
#    elif v.is_placeholder:
#        pass
#    else:
#        print('Need to evaluate Function %s' % v.owner)

#myinput = find_by_name(model, "myinput", depth=-1)
#data = np.arange(165, dtype=np.float32).reshape((165,))
#set_trace()

#g = ExpressionGenerator()
#g.generate(model, set(), None)
#print("Listing:\n%s" % g.full_listing)

#nodes = depth_first_search(model, lambda x: True, depth=-1)

#for n in nodes:
#   if isinstance(n, Parameter):
#       print('Parameter, name %s, shape %s, uid %s' % (n.name, n.shape, n.uid))
#   elif isinstance(n, Constant):
#       print('Constant, name %s, shape %s, uid %s' % (n.name, n.shape, n.uid))
#   elif isinstance(n, Variable):
#       print('Variable, name %s, shape %s, uid %s, is input %s' % (n.name, n.shape, n.uid, str(n.is_input)))
#   elif isinstance(n, Function):
#       print('Function, name %s, uid %s' % (n.name, n.uid))
#       if n.is_primitive:
#           print('	Operation %s' % n.op_name)
#       for i in n.inputs:
#           print('	Inputs, name %s, uid %s' % (i.name, i.uid))
#       for o in n.outputs:
#           print('	Outputs, name %s, uid %s' % (o.name, o.uid))
#       for p in n.parameters:
#           print('	Parameters, name %s, uid %s' % (p.name, p.uid))
#   else:
#       print('Name %s, type %s' % (n.name, type(n)))

#filename = 'x'
#plot(model, filename + '.pdf')
#for n in nodes:
#   if isinstance(n, Function) and n.is_block:
#       print(n.uid)
#       plot(n.block_root, filename + n.uid + '.pdf')
#       mapping = n.block_arguments_mapping
#       for comp, actual in mapping:
#           print('	%s -> %s' % (comp.uid, actual.uid))

