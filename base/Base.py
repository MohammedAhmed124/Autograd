
from ..utils import ContextObject
import numpy as np



class BaseArray(np.ndarray):
    def __new__(cls, input_array , requires_grad:bool = False ,**kwargs):
        # Convert input_array to an ndarray instance
        obj = np.asarray(input_array , **kwargs).view(cls)
        obj.requires_grad = requires_grad
        obj.is_grad_container=False
        obj.grad = np.zeros_like(obj) if requires_grad else None

        obj.grad_fn = None
        obj._is_leaf = True


        if obj._is_leaf and obj.requires_grad:
            obj.grad.is_grad_container = True
        return obj
    


    def __array_finalize__(self, obj): #self is the object that is created and obj is what you inherite from
        if obj is None: return
        self.requires_grad = getattr(obj , "requires_grad" , False)
        self.grad = None
        self.grad_fn = None
        self._is_leaf = False

    


    def __repr__(self):
        rep = super(BaseArray , self).__repr__()[0:-1]
        requires_grad = getattr(self , "requires_grad" , False)
        rep+= f", requires_grad={requires_grad}"
        rep+= f", grad_fn={getattr(self.grad_fn , "_get_name" , lambda: None)()})" if requires_grad else ")"
        return rep
    def __str__(self):
        return self.__repr__()


    def _set_next_object_attributes( self , other , result_obj , grad_fn = None , grad_fn_kwargs=None ):
         if not grad_fn_kwargs:
            grad_fn_kwargs = {}
         result_obj.grad_fn = grad_fn(ABackward = self.grad_fn if isinstance(self , BaseArray) else None
                                    ,BBackward = other.grad_fn if isinstance(other , BaseArray) else None
                                    ,ctx = ContextObject(self , other)
                                    ,**grad_fn_kwargs)
         result_obj._is_leaf = False
         result_obj.requires_grad = True

    def _is_scaler(self):
        return True if self.shape==() else False
    

    def _is_grad_container(self):
        """Utility to check if the tensor has a grad container (stores gradients)."""
        return getattr(self , "is_grad_container" , False)








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



