from collections import OrderedDict
from numpy import zeros

class BaseFrame():

    ''' Base Frame Data-Model Class'''

    valid_node_types = ['INPUT', 'SIMPLE_SUM']

    def __init__(self):
        inputs = zeros((1))
        outputs = zeros((1))
        self.layers = OrderedDict()
        self.layers['input'] = inputs
        self.layers['output'] = outputs
        self.layer_types = OrderedDict()
        self.layer_types['input'] = 'INPUT'
        self.layer_types['output'] = 'SIMPLE_SUM'
        self.layer_count = 0
        self.frame = []
        self.frame.append(inputs.size)
        self.frame.append(outputs.size)

        self.node_dictionary = OrderedDict()
        for indx, node_type in self.valid_node_types:
            self.node_dictionary[str(indx + 1)] = node_type

    def addValidNodeType(self, node_type):
        ''' Adds valid node type to data class. '''
        node_type = node_type.upper()
        if node_type in self.valid_node_types:
            return
        self.valid_node_types.append(node_type)
        self.node_dictionary[str(len(self.valid_node_types))] = node_type

    def nodeTypeCheck(self, node_type):
        ''' Checks if Node Type is Valid'''
        valid_node_types = ''
        for indx, valid_node_type in enumerate(self.valid_node_types):
            if isinstance(node_type, str) and node_type.upper() == valid_node_type.upper():
                return
            if indx != (len(self.valid_node_types) - 1):
                valid_node_type = valid_node_type + ', '
            else:
                valid_node_type = 'and ' + valid_node_type
            valid_node_types += valid_node_type
        raise ValueError('node_type must be string of valid node type.\
            \n    Ex.)\n        %s'%valid_node_types)

    def addHiddenLayer(self, rows, position=-1, node_type='SIGMOID', dtype='float64'):
        ''' Add Hidden Layer to frame. '''
        try:
            rows = int(rows)
            position = int(position)
        except ValueError:
            raise ValueError('addHiddenLayer Method requires integer inputs.')
        node_type = node_type.upper()
        self.nodeTypeCheck(node_type)
        nodes = zeros((rows), dtype=dtype)
        self.layers[str(self.layer_count)] = nodes
        self.layer_types[str(self.layer_count)] = node_type
        self.layer_count += 1
        if position == 0:
            position = 1
        elif position == len(self.frame):
            position = -1
        self.frame.insert(position, nodes.size)

    def resetLayer(self, rows, layer_title, node_type='SIGMOID', dtype='float64'):
        ''' Base Layer Reset. '''
        node_type = node_type.upper()
        if node_type is not self.layer_types['%s'%layer_title]:
            self.nodeTypeCheck(node_type)
            self.layer_types['%s'%layer_title] = node_type
        if rows is not None:
            if not isinstance(rows, int):
                raise TypeError('%sResetLayer rows must both be left undefined\
                    \n or both be defined as an integer.'%layer_title)
            nodes = zeros((rows), dtype=dtype)
            self.layers['%s'%layer_title] = nodes
            return
        rows = self.layers['%s'%layer_title].size
        nodes = zeros((rows), dtype=dtype)
        self.layers['%s'%layer_title] = nodes

    def inputResetLayer(self, rows):
        ''' Resets Input Layer; may be used to redimension inputs. '''
        self.resetLayer(rows, layer_title='input', node_type='INPUT')

    def outputResetLayer(self, rows, node_type='SIMPLE_SUM'):
        ''' Resets Output Layer; may be used to redimension inputs. '''
        self.resetLayer(rows, layer_title='output', node_type=node_type)

    def addNodeToLayer(self, layer_title, number_of_new_nodes):
        ''' Add Node Type to Hidden Layer. '''
        try:
            layer_title = int(layer_title)
            number_of_new_nodes = int(number_of_new_nodes)
        except ValueError:
            raise ValueError('addNodeToLayer Method requires integer layer_title and integer number_of_new_nodes.')
        rows = self.layers['%s'%layer_title].size
        nodes = zeros((rows + number_of_new_nodes))
        for row in range(rows):
            nodes[row] = self.layers[str(layer_title)][row]
        self.layers[str(layer_title)] = nodes
        self.frame[layer_title] = nodes.size
