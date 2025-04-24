from typing import *
import numpy as np
# from  .tensor import tensor

class ContextObject:
    #:Union[tensor , int , float]
    def __init__(self , a , b:Union[np.ndarray , int , float]):
        self.a = a
        self.b = b
    def _a_requires_grad(self):
        return getattr(self.a , "requires_grad" , False)
    def _b_requires_grad(self):
        return getattr(self.b , "requires_grad" , False)
