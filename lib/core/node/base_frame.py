from collections import OrderedDict
from numpy import zeros

class BaseFrame():

    ''' Base Frame Data-Model Class'''

    def __init__(self):
        inputs = zeros((1, 1))
        outputs = zeros((1, 1))
        self.layers = OrderedDict()
        self.layers['input'] = inputs
        self.layers['output'] = outputs
        self.layer_count = 0
        self.frame = []
        self.frame.append(inputs.shape)
        self.frame.append(outputs.shape)

    def addHiddenLayer(self, depth, width=1, position=-1):
        ''' Add Hidden Layer to frame. '''
        try:
            depth = int(depth)
            width = int(width)
            position = int(position)
        except ValueError:
            raise ValueError('addHiddenLayer Method requires integer inputs.')
        nodes = zeros((depth, width))
        self.layers[str(self.layer_count)] = nodes
        self.layer_count += 1
        if position == 0:
            position = 1
        elif position == len(self.frame):
            position = -1
        self.frame.insert(position, nodes.shape)

    def resetLayer(self, depth, width, frame_layer, layer_title):
        ''' Base Layer Reset. '''
        if depth is not None and width is not None:
            layer = zeros((depth, width))
            self.layers['%s'%layer_title] = layer
            self.frame[frame_layer] = (depth, width)
            return
        elif depth is not width:
            raise TypeError('%sResetLayer depth and width must both be left undefined\
                \n or both be defined as an integer.'%layer_title)
        self.layers['%s'%layer_title] = zeros((self.layers['%s'%layer_title].shape))

    def inputResetLayer(self, depth=None, width=None):
        ''' Resets Input Layer; may be used to redimension inputs. '''
        self.resetLayer(depth, width, frame_layer=0, layer_title='input')

    def outputResetLayer(self, depth=None, width=None):
        ''' Resets Output Layer; may be used to redimension inputs. '''
        self.resetLayer(depth, width, frame_layer=-1, layer_title='output')
