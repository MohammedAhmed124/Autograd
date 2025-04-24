from ..base import BaseBackwardFunction
import numpy as np
class SumBackward(BaseBackwardFunction):
    def __init__(self , axis , keepdims=False , **kwargs):
        super(SumBackward , self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims
    def __call__(self , grad_output):
        #y = x1 * x2
        #∂y/∂x1 = x2
        #∂y/∂x2 = x1
        input_shape = self.ctx.a.shape
        keepdims = self.keepdims
        axis = self.axis
        if not keepdims:
            if isinstance(axis , int):
                grad_output = np.expand_dims(grad_output , axis)
        

        if not axis:
            grad_a = grad_output*np.ones_like(self.ctx.a ) if self.ctx._a_requires_grad() else None
            grad_b = grad_output*np.ones_like(self.ctx.a) if self.ctx._b_requires_grad() else None
        
        elif isinstance(axis , int):
            grad_a = np.broadcast_to(grad_output , input_shape) if self.ctx._a_requires_grad() else None
            grad_b =  np.broadcast_to(grad_output , input_shape) if self.ctx._b_requires_grad() else None


        return grad_a , grad_b


