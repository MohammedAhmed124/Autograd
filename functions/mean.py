from ..base import BaseBackwardFunction , BaseArray
import numpy as np



class MeanBackward(BaseBackwardFunction):
    def __init__(self , axis , keepdims ,**kwargs):
        super(MeanBackward , self).__init__(**kwargs)
        self.axis = axis 
        self.keepdims = keepdims
    def __call__(self , grad_output):
        if not self.ctx._a_requires_grad():
            return (None , None)
        input_ = self.ctx.a
        axis = self.axis
        keepdims = self.keepdims
        if axis is None:
            grad_output = np.ones_like(input_)*grad_output
        else:
        
            if isinstance(axis , int):
                axis = (axis ,)

            if not keepdims:
                for ax in sorted(axis):
                   
                    grad_output = np.expand_dims(grad_output , ax=ax)

            grad_output = np.broadcast_to(grad_output , input_.shape)

            axis_mul = 1
            for ax in axis:
                axis_mul*=input_.shape[ax]

            grad_output = grad_output/axis_mul


        return (grad_output , None)




