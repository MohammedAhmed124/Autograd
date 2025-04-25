from typing import *
import numpy as np
from functools import wraps 



def _any_requires_grad(*args):
    """Utility to check if any of the tensors in the args require gradients."""
    return any(getattr(x, 'requires_grad', False) for x in args)



class ContextObject:
    def __init__(self , a , b:Union[np.ndarray , int , float]):
        self.a = a
        self.b = b
    def _a_requires_grad(self):
        return getattr(self.a , "requires_grad" , False)
    def _b_requires_grad(self):
        return getattr(self.b , "requires_grad" , False)


def block_in_place_autograd(in_place_func):
    @wraps(in_place_func)
    def wrapper(self , other ,*args, **kwargs):
        if  _any_requires_grad(self, other):
            if not self._is_grad_container():
                raise ValueError("in-place modification to variables which require gradients hurts the computational graph.\
                                avoid in-place modifications for tensors which takes place in the computational graph")
        
        return in_place_func(self , other , *args , **kwargs)
    return wrapper


