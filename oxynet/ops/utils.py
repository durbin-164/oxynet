def handle_array_broadcasting(grad, data):
    ndims_added = grad.ndim - data.ndim

    for _ in range(ndims_added):
        grad = grad.sum(axis = 0)

    for i, dim in enumerate(data.shape):
        if dim == 1:
            grad = grad.sum(axis = i, keepdims = True)

    return grad

def inv_permutation(permutation):
    """Get the inverse of a permutation. Used to invert a transposition for example.

    Args:
        permutation (list or tuple): permutation to invert.

    Returns:
        list
    """
    inverse = [0] * len(permutation)
    for i, p in enumerate(permutation):
        inverse[p] = i
    return inverse