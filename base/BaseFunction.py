from .BaseArray import  BaseArray
from ..utils import ContextObject


class BaseBackwardFunction(object):
    _class_count = {}
    def __init__(self  , next_functions = (None , None), ctx:ContextObject = None):
        if not isinstance(next_functions , tuple):
            raise TypeError("next_functions should be passed as a tuple")
        self.next_functions = next_functions
        self.ctx = ctx
    def __new__(cls , *args, **kwargs):
        if not cls in cls._class_count:
            cls._class_count[cls] = 0
        cls._class_count[cls]+=1
        return super().__new__(cls)
    

    def _get_name(self):
        return self.__class__.__name__ + str(BaseBackwardFunction._class_count.get(self.__class__, None))
     
    def _update_if_leaf(self , grad_a , grad_b):
        if getattr(self.ctx.a , "_requires_grad" , False):
            if getattr(self.ctx.a , "_is_leaf", False) and isinstance(grad_a ,BaseArray):
                self.ctx.a.grad += grad_a
        if getattr(self.ctx.b , "_requires_grad" , False):
            if getattr(self.ctx.b , "_is_leaf", False) and isinstance(grad_b ,BaseArray):
                self.ctx.b.grad += grad_b



