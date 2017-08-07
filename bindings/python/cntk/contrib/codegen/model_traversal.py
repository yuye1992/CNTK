from cntk import *
from cntk import cntk_py
from cntk.logging import *
import cntk.variables
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pdb import set_trace
import itertools

model_path = r'c:\repo\halide_playground\my_super.model'

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

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
    for node in g.nodes():
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
    for node in g.nodes():
        successors = g.successors(node)
        for successor in successors:
            label = node.name if node.name else node.uid # the Output variables have no name if the function has none
            label += '\n' + shape_desc(node) + '\n' + node_desc(node)

            dot_object.add_edge(pydot.Edge(dot_nodes[node.uid], dot_nodes[successor.uid], label=label))

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
            self._convert(g, output, None, set(), {})
        return g

    def _convert(self, g, node, child, visited, placeholder_mapping):
        from cntk import cntk_py
        is_function = isinstance(node, cntk_py.Function)

        # First thing - add an edge between the child and the node
        # skipping blocks if needed
        if child is not None:
            g.add_edge(
                dict(node.block_outputs_mapping)[child] if is_function and node.is_block else node,
                child) 

        if node.uid in visited:
            return
        visited.add(node.uid)

        if is_function:
            if node.is_block:
                placeholder_mapping.update(node.block_arguments_mapping)
                outputs_mapping = dict(node.block_outputs_mapping)
                inner_output_variable = outputs_mapping[child]
                self._convert(g, inner_output_variable, child, visited, placeholder_mapping)
            elif node.is_primitive:
                for i in node.inputs:
                    i = placeholder_mapping[i] if i.is_placeholder else i
                    self._convert(g, i, node, visited, placeholder_mapping)
            else:
                set_trace()
                raise ValueError("Unexpected function node type %s" % node)

        elif node.is_parameter or node.is_constant or node.is_input:
            pass
        elif node.is_output:
            self._convert(g, node.owner, node, visited, placeholder_mapping)
        elif node.is_placeholder:
            actual_node = placeholder_mapping[node]
            self._convert(g, actual_node, visited, placeholder_mapping)
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
        for node in g.nodes():
            if not (isinstance(node, cntk.variables.Variable) and node.is_output):
                continue
     
            successors = g.successors(node)
            if len(successors) == 0:
                continue
     
            predecessors = g.predecessors(node)      
            if len(predecessors) == 0:
                raise ValueError("Unexpected output node with no ancestors")
     
            for p, s in itertools.product(predecessors, successors):
                g.add_edge(p, s, data = node, label = node.uid)
     
            g.remove_node(node)
            removed = True

def split_past_values(g):
    for node in g.nodes():
        if not isinstance(node, cntk_py.Function):
            continue
        if node.op_name != 'PastValue':
            continue

        external_output = cntk.output_variable(dynamic_axes=node.dynamic_axes, shape=node.shape, dtype=node.dtype, name=node.uid + '_external_output')
        external_input = cntk.input_variable(dynamic_axes=node.dynamic_axes, shape=node.shape, dtype=node.dtype, name=node.uid + '_external_input')

        g.add_node(external_input, original=node, name=external_input.uid)
        g.add_node(external_output, original=node, name=external_output.uid)

        for successor in g.successors(node):
            g.add_edge(external_input, successor)

        for predecessor in g.predecessors(node):
            g.add_edge(predecessor, external_output)

        g.remove_node(node)

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

class ExpressionGenerator:
    def __init__(self):
        super(ExpressionGenerator, self).__init__()

    def generate(self, nodes):
        from cntk import cntk_py

        for node in nodes:
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
        super(HalideExpressionGenerator, self).__init__()
        self.uid_to_expression = {}
        self.graph = g
        self.listing = ''

    def generate_value(self, node):
        if node.dtype == np.float32:
            data_type = 'float'
        else:
            assert node.dtype == np.float64
            data_type = 'double'
        expression = 'const %s* c_%s = {' % (data_type, node.uid)        
        for value in node.value.flatten():
            expression += ('%f' % value) + ', '
        expression += '};\n'
        if len(node.shape) == 2:
            expression += 'Buffer<%s> %s = Buffer<%s>(c_%s, %s, %s, "%s")' % (data_type, node.uid, data_type, node.uid, node.shape[0], node.shape[1], node.uid)
        elif len(node.shape) == 1:
            expression += 'Buffer<%s> %s = Buffer<%s>(c_%s, %s, "%s")' % (data_type, node.uid, data_type, node.uid, node.shape[0], node.uid)
        else:
            raise ValueError('Unexpected shape encoutered, only 1 and 2D are currently supported %s' % node)
        return expression

    def generate_parameter(self, node):
        node = node.as_parameter()
        exp = self.generate_value(node)
        self.listing += exp + '\n'

    def generate_constant(self, node):
        node = node.as_constant()
        exp = self.generate_value(node)
        self.listing += exp + '\n'

    def generate_input(self, node):
        set_trace()

        pass

    def generate_output(self, node):
        operands = self.graph.predecessors(node)
        arguments = '%s' % operands[0].uid
        for o in operands[1:]:
            arguments += ', %s' % o.uid
        exp = '%s(%s)' % (node.uid, arguments)
        self.listing += exp;

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
        self.uid_to_expression[node.uid] = '%s' % node.uid

    def generate_binary_call(self, node, op_name):
        operands = self.graph.predecessors(node)
        if len(operands) != 2:
            raise ValueError('Operation "%s" expects 2 operands, given %s', op_name, str(operands))
        exp = '%s = %s(%s, %s)' % tuple([node.uid, op_name] + [o.uid for o in operands])
        self.listing += exp + '\n'

    def generate_binary_op(self, node, op_name):
        operands = self.graph.predecessors(node)
        if len(operands) != 2:
            raise ValueError('Operation "%s" expects 2 operands, given %s', op_name, str(operands))
        exp = '%s = %s %s %s' % tuple([node.uid, operands[0], op_name, operands[1]])
        self.listing += exp + '\n'

    def generate_unary_op(self, node, op_name):
        operands = self.graph.predecessors(node)
        if len(operands) != 1:
            raise ValueError('Operation "%s" expects 1 operand, given %s', op_name, str(operands))
        exp = '%s = %s(%s)' % tuple([node.uid, op_name, operands[0]])
        self.listing += exp + '\n'

    def generate_times(self, node):
        self.generate_binary_call(node, 'Times')

    def generate_element_times(self, node):
        self.generate_binary_op(node, '*')

    def generate_plus(self, node):
        self.generate_binary_op(node, '+')

    def generate_stable_sigmoid(self, node):
        self.generate_unary_op(node, 'Sigmoid')

    def generate_tanh(self, node):
        self.generate_unary_op(node, 'Tanh')

    def generate_slice(self, node):
        operand = self.graph.predecessors(node)
        if len(operand) != 1:
            raise ValueError('Operation "slice" expects 2 operands')
        operand = operand[0]
        if len(operand.shape) == 1:
            begin = node.attributes['beginIndex']
            end = node.attributes['endIndex']
            stride = node.attributes['sliceStrides']
            if stride != 1:
                raise ValueError('Unexpected stride "%s", only stride of 1 is currently supported' % str(stride))
            exp = '%s = ((%s)(%d, %d))' % (node.uid, operand.uid, begin, end)
        else:
            raise ValueError('Slice is not supported on node of shape %s' % str(node.shape)) 
        self.listing += exp + '\n'

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################



model = Function.load(model_path)

set_trace()
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
    print('Node name %s, uid %s' % (node.name, node.uid))

generator = HalideExpressionGenerator(g)
generator.generate(nodes_sorted_for_generation)

listing = generator.listing

with open('result', 'w') as f:
    f.write(listing)

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

