from ..base import BaseBackwardFunction
import numpy as np


class SubBackward(BaseBackwardFunction):
    def __call__(self , grad_output):
        # y = a-b
        #dy/da = 1
        #dy/db = -1
        # grad_a = np.ones_like(self.ctx.a) if self.ctx._a_requires_grad() else None
        # grad_b = -1*np.ones_like(self.ctx.a) if self.ctx._a_requires_grad() else None

        return (grad_output*1 , grad_output*-1 )

