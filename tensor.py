import numpy as np
from .functions import *
from .utils import ContextObject ,block_in_place_autograd
from .base import BaseArray , BaseBackwardFunction
from .context import no_grad
from .config import GlobalConfig

from .utils import _any_requires_grad


class tensor(BaseArray):
    def __add__(self , other):
        if not isinstance(other, tensor):
            other = tensor(other)
        result_obj = super(tensor , self).__add__(other)
        if  _any_requires_grad(self, other) and not GlobalConfig.is_propagating_backwards():
            self._set_next_object_attributes( other , result_obj , grad_fn = addBackward)
        

        return result_obj
    def __mul__(self , other):
        if not isinstance(other, tensor):
            other = tensor(other)
        result_obj = super(tensor , self).__mul__(other)
        if  _any_requires_grad(self, other) and not GlobalConfig.is_propagating_backwards():
            self._set_next_object_attributes( other , result_obj , grad_fn = MulBackward)
        

        return result_obj
    # def __div__(self , other):
    #     return super(tensor , self).__div__(other)
    
    def __matmul__(self , other):
        if not isinstance(other, tensor):
            other = tensor(other)
        result_obj = super(tensor , self).__matmul__(other)
        if  _any_requires_grad(self, other) and not GlobalConfig.is_propagating_backwards():
            self._set_next_object_attributes( other , result_obj , grad_fn = matmulBackward)
        return result_obj
    
    def transpose(self):
        result_obj = super(tensor ,self).transpose()
        if  _any_requires_grad(self, None) and not GlobalConfig.is_propagating_backwards():
             self._set_next_object_attributes( other = None , result_obj= result_obj, grad_fn = TransposeBackward )

        return result_obj
    

    def sum(self , axis = None , keepdims =False , **kwargs):
        result_object =  super(tensor , self).sum(axis = axis, keepdims =keepdims , **kwargs)
        if  _any_requires_grad(self, None) and not GlobalConfig.is_propagating_backwards():
            grad_fn = SumBackward
            grad_fn_kwargs = {"axis" : axis ,
                               "keepdims" :keepdims}
            self._set_next_object_attributes( other=None , result_obj=result_object , grad_fn =grad_fn , grad_fn_kwargs=grad_fn_kwargs )
        
        return result_object
    
    def mean(self , axis=None, keepdims=False, **kwargs):
        result_object =  super(tensor , self).mean(axis = axis, keepdims =keepdims , **kwargs)
        if  _any_requires_grad(self, None) and not GlobalConfig.is_propagating_backwards():
            grad_fn = MeanBackward
            grad_fn_kwargs = {"axis" : axis ,
                               "keepdims" :keepdims}
            self._set_next_object_attributes( other=None , result_obj=result_object , grad_fn =grad_fn , grad_fn_kwargs=grad_fn_kwargs )
        
        return result_object


    
    @property
    def T(self):
        return self.transpose()


    
    @block_in_place_autograd
    def __iadd__(self , other):
        return super(tensor , self).__iadd__(other)
    

    @block_in_place_autograd
    def __imul__(self , other):
        return super(tensor , self).__imul__(other)
    @block_in_place_autograd
    def __imatmul__(self ,other):
        return super(tensor , self).__imatmul__(other)
    

        

    def backward(self):
        if not self._is_scaler():
            raise ValueError("backward is only called on scaler values")
        
        if not self.requires_grad:
            raise ValueError("root tensor does not require grad")

        
        with no_grad():
            def recurse_backwards( grad_fn , grad_output = 1):
                grad_a , grad_b = grad_fn(grad_output=grad_output)
                grad_fn._update_if_leaf(grad_a , grad_b)
                if isinstance(grad_fn.ABackward , BaseBackwardFunction) :
                    recurse_backwards( grad_fn.ABackward , grad_output = grad_a)
                if isinstance(grad_fn.BBackward , BaseBackwardFunction) :
                    recurse_backwards(grad_fn.BBackward , grad_output = grad_b)

            recurse_backwards( self.grad_fn , grad_output = 1)
    
        


    

    
    
