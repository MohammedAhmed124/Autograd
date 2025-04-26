from ..base import BaseBackwardFunction
import numpy as np

class TransposeBackward(BaseBackwardFunction):

    def __call__(self , grad_output):
        return grad_output.T , None
        