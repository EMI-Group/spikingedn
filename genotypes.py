from collections import namedtuple

Genotype = namedtuple('Genotype', 'cell cell_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

PRIMITIVES1 = [
    'skip_connect',
    'convBR_3x3',
    'convBR_5x5',
]

PRIMITIVES2 = [
    'skip_connect',
    'convBR_5x5',
]


