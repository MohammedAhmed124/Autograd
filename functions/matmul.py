from ..base import BaseBackwardFunction
import numpy as np


class matmulBackward(BaseBackwardFunction):
    def __call__(self , grad_output):
        a = self.ctx.a
        b = self.ctx.b

        #y = x1 matmul x2
        #∂y/∂x1 = x2 
        #∂y/∂x2 = x1 

        grad_a = grad_output @ b.T  if self.ctx._a_requires_grad() else None
        grad_b = a.T @ grad_output if self.ctx._b_requires_grad() else None

        return grad_a , grad_b
