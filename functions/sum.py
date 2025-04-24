from ..base import BaseBackwardFunction
import numpy as np
class SumBackward(BaseBackwardFunction):
    def __call__(self , grad_output):
        #y = x1 * x2
        #∂y/∂x1 = x2
        #∂y/∂x2 = x1

        grad_a = grad_output*np.ones_like(self.ctx.a ) if self.ctx._a_requires_grad() else None
        grad_b = grad_output*np.ones_like(self.ctx.a) if self.ctx._b_requires_grad() else None

        return grad_a , grad_b
