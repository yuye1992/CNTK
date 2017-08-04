from cntk import *
from cntk.logging import *
import cntk.variables
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pdb

model_path = r'c:\repo\halide_playground\my_super.model'

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

        g = nx.DiGraph()
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
                    i = placeholder_mapping[i] if i.is_placeholder() else i
                    self._convert(g, i, node, visited, placeholder_mapping)
            else:
                pdb.set_trace()
                raise ValueError("Unexpected function node type %s" % node)

        elif node.is_parameter or node.is_constant or node.is_input:
            pass
        elif node.is_output:
            self._convert(g, node.owner, node, visited, placeholder_mapping)
        elif node.is_placeholder():
            actual_node = placeholder_mapping[node]
            self._convert(g, actual_node, visited, placeholder_mapping)
        else:
            pdb.set_trace()
            raise ValueError("Unexpected node type %s" % node)


class ExpressionGenerator:
    def __init__(self):
        super(ExpressionGenerator, self).__init__()
        self.variable_table = {}
        self.full_listing = ''

    def generate(self, node, visited, child):
        if node.uid in visited:
           return
        visited.add(node.uid)

        from cntk import cntk_py

        if isinstance(node, cntk_py.Function):
            if node.is_block:
                new_root = node.block_root
                new_root = new_root.root_function if new_root.is_composite else new_root
                mapping = node.block_arguments_mapping
                for placeholder, actual_input in mapping:
                     self.generate(actual_input, visited, child)
                self.generate(new_root, visited, child)
            elif node.is_primitive:
                for p in node.inputs:
                    self.generate(p, visited, node)
                self.generate_primitive_function(node)
            elif node.is_composite:
                self.generate(node.root_function, visited, child)
            else:
                pdb.set_trace()
                raise ValueError("Unexpected function node")
        elif node.is_parameter:
            self.generate_parameter(node)
        elif node.is_constant:
            self.generate_constant(node)
        elif node.is_input:
            self.generate_input(node)
        elif node.is_output:
            self.generate(node.owner, visited, node)
            self.generate_output(node)
        elif node.is_placeholder:
            self.generate_placeholder(node)
        else:
            pdb.set_trace()
            raise ValueError("Unexpected node type")

    def generate_parameter(self, node):
        self.full_listing += "Parameter() { name : %s, uid : %s, shape : %s }\n" % (node.name, node.uid, node.shape)

    def generate_constant(self, node):
        self.full_listing += "Constant() { name : %s, uid : %s, shape : %s }\n" % (node.name, node.uid, node.shape)

    def generate_input(self, node):
        self.full_listing += "Input() { name : %s, uid : %s, shape : %s }\n" % (node.name, node.uid, node.shape)

    def generate_output(self, node):
        self.full_listing += "Output() { name : %s, uid : %s, shape : %s }\n" % (node.name, node.uid, node.shape)

    def generate_primitive_function(self, node):
        self.full_listing += "Primitive Function() { name : %s, uid : %s, op name : %s }\n" % (node.name, node.uid, node.op_name)

    def generate_placeholder(self, node):
        self.full_listing += "Placeholder() { name : %s, uid : %s }\n" % (node.name, node.uid)
      
model = Function.load(model_path)

pdb.set_trace()
c = ModelToGraphConverter()
g = c.convert(model)


# Let's put output variables on the edges instead of nodes.
def remove_output_node(g):
    for node in g.nodes():
        if isinstance(node, cntk.variables.Variable) and node.is_output:
            successors = g.successors(node)
            if len(successors) == 0:
                continue

            ancestors = []      
            for other_node in g.nodes():
                if other_node.uid == node.uid:
                    continue
                if node in g.adj[other_node]:
                    ancestors.append(other_node)

            if len(ancestors) == 0:
                pdb.set_trace()
                raise ValueError("Unexpected output node with no ancestors")

            for child in successors:
                for ancestor in ancestors:
                    g.add_edge(ancestor, child, data = node)

            g.remove_node(node)
            return True
    return False

def remove_output_nodes(g):
   while remove_output_node(g):
       pass


remove_output_nodes(g)

for node in g.nodes():
    print('Node id %s' % node.uid)
    for i in g.adj[node]:
        print('       Connected to %s' % i.uid)

#nx.draw(g)
#plt.show()

#outputs = { o for o in model.outputs }
#vars_in_eval_order = model.get_evaluation_order(outputs)


#for v in vars_in_eval_order:
#    import pdb; pdb.set_trace()
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
#import pdb; pdb.set_trace()

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

