from ..base import BaseBackwardFunction
import numpy as np


class SumBackward(BaseBackwardFunction):
    def __init__(self , axis , keepdims=False , **kwargs):
        super(SumBackward , self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims
    def __call__(self , grad_output):
        input_shape = self.ctx.a.shape
        keepdims = self.keepdims
        axis = self.axis


        if axis is None:
            grad_output = np.ones_like(self.ctx.a) * grad_output

        else:
            # Turn int axis to tuple
            if isinstance(axis, int):
                axis = (axis,)

            if not keepdims:
                # Insert singleton dims at reduced axes
                for ax in sorted(axis):
                    grad_output = np.expand_dims(grad_output, axis=ax)



            grad_output = np.broadcast_to(grad_output, input_shape)


        grad_a = grad_output if self.ctx._a_requires_grad() else None
        grad_b = grad_output if self.ctx._b_requires_grad() else None

        # print("inside sum")

        # print(grad_a)


        return grad_a, grad_b



