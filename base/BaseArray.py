
from ..utils import ContextObject
import numpy as np



class BaseArray(np.ndarray):
    def __new__(cls, input_array , requires_grad:bool = False ,**kwargs):
        # Convert input_array to an ndarray instance
        obj = np.asarray(input_array , **kwargs).view(cls)
        obj.requires_grad = requires_grad
        obj.is_grad_container=False
        obj._is_leaf = True
        
        obj.grad = np.zeros_like(obj) if (requires_grad and obj._is_leaf) else None

        obj.grad_fn = None


        if obj._is_leaf and obj.requires_grad:
            obj.grad.is_grad_container = True

        if obj.requires_grad and np.isnan(obj).sum()!=0:
            raise ValueError("A tensor which requires gradient calculations should not contain NaNs")
        return obj
    


    def __array_finalize__(self, obj): #self is the object that is created and obj is what you inherite from
        if obj is None: 
            return
        
        self.requires_grad = getattr(obj , "requires_grad" , False)
        self.grad = None
        self.grad_fn = None
        self._is_leaf = False



    def __array_function__(self, func, types, args, kwargs):
        # Intercept np function calls and return an instance of MyTensor
        result =   super(BaseArray , self).__array_function__(func, types, args, kwargs)
        if not isinstance(result, self.__class__):
            return self.__class__(result)  # Ensure the result is of MyTensor type
        return result
    



    


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






