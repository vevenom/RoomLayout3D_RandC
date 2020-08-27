from collections import namedtuple

# Struct that describes the layout plane
LayoutPlane = namedtuple('LayoutPlane',
                           ['plane',
                            'mask',
                            'type',
                            ])
LayoutPlane.__new__.__defaults__ = (None,) * len(LayoutPlane._fields)

# Struct that describes the candidate layout component
CandidateLayoutComp = namedtuple('CandidateLayoutComp',
                           ['plane',
                            'mask',
                            'type',
                            'poly',
                            'poly_mask',
                            'cost',
                            'area'
                            ])
LayoutPlane.__new__.__defaults__ = (None,) * len(LayoutPlane._fields)