def flatten_tuple(nested_tuple):
    for item in nested_tuple:
        if isinstance(item, (tuple, list)):
            yield from  flatten_tuple(item)
        else:
            yield item