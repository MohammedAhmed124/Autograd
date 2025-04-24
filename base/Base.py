
from ..utils import ContextObject
import numpy as np



class BaseArray(np.ndarray):
    def __new__(cls, input_array , requires_grad:bool = False ,**kwargs):
        # Convert input_array to an ndarray instance
        obj = np.asarray(input_array , **kwargs).view(cls)
        obj.requires_grad = requires_grad

        obj.grad = np.zeros_like(obj) if requires_grad else None
        obj.grad_fn = None
        obj._is_leaf = True
        return obj
    


    def __repr__(self):
        rep = super(BaseArray , self).__repr__()[0:-1]
        requires_grad = getattr(self , "requires_grad" , False)
        rep+= f", requires_grad={requires_grad}"
        rep+= f", grad_fn={getattr(self.grad_fn , "_get_name" , lambda: None)()})" if requires_grad else ")"

        return rep


    def _set_next_object_attributes( self , other , result_obj , grad_fn = None):
         result_obj.grad_fn = grad_fn(ABackward = self.grad_fn if isinstance(self , BaseArray) else None
                                    ,BBackward = other.grad_fn if isinstance(other , BaseArray) else None
                                    ,ctx = ContextObject(self , other))
         result_obj._is_leaf = False
         result_obj.requires_grad = True







class BaseBackwardFunction(object):
    _class_count = {}
    def __init__(self  , ABackward = None, BBackward = None , ctx:ContextObject = None):
        self.ABackward = ABackward
        self.BBackward = BBackward
        self.ctx = ctx
    def __new__(cls , *args, **kwargs):
        if not cls in cls._class_count:
            cls._class_count[cls] = 0
        cls._class_count[cls]+=1
        return super().__new__(cls)
    

    def _get_name(self):
        return self.__class__.__name__ + str(BaseBackwardFunction._class_count.get(self.__class__, None))
     
    def _update_if_leaf(self , grad_a , grad_b):
        if getattr(self.ctx.a , "_is_leaf", False) and isinstance(grad_a ,BaseArray):
            self.ctx.a.grad += grad_a

        if getattr(self.ctx.b , "_is_leaf", False)and isinstance(grad_b ,BaseArray):
            self.ctx.b.grad += grad_b



