import itertools
# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
# For sweeping
def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def grid_search(sweep_space):
    grid_dict = {}
    for k in sweep_space:
        if 'feasible_points' in sweep_space[k]:
            grid_dict[k] = sweep_space[k]['feasible_points']
        else:
            raise NotImplementedError
    return list(dict_product(grid_dict))