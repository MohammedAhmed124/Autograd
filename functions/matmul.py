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

        # print(f"grad_output_a = grad_output * b.T\n" , grad_output , "\n" , b.T , "\n" , grad_a , end = "\n\n\n\n")

        # print(f"grad_output_b = a.T @ grad_output\n" , a.T , "\n" ,  grad_output  ,"\n" , grad_b , end = "\n\n\n\n")

        return grad_a , grad_b
