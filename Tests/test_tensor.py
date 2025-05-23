import numpy as np
import torch
import pytest
from ..tensor import tensor
import sys
import os
from collections import OrderedDict


from ..tensor import tensor


@pytest.mark.parametrize("case", range(10000))
def test_tensor_function(case):
    requires_grad_a = bool(np.random.randint(0 , 2))
    requires_grad_b = bool(np.random.randint(0 , 2))
    x = tensor([[1 , 2 , 3] , [1 , 1 , 2]] , requires_grad= requires_grad_a , dtype = np.float64 )
    b = tensor([[2 , 3 , 1] , [3 , 24 , 2]]  , requires_grad=requires_grad_b ,  dtype = np.float64)

    x_torch = torch.tensor([[1 , 2 , 3] , [1 , 1 , 2]] , requires_grad= requires_grad_a , dtype=torch.float64 )
    b_torch = torch.tensor([[2 , 3 , 1] , [3 , 24 , 2]]  , requires_grad=requires_grad_b ,  dtype=torch.float64)
    x_ref , b_red = x , b

    x_torch_ref , b_torch_ref = x_torch , b_torch


    int_ = OrderedDict()
    bucket_1 = [ x , b]

    bucket_2 = [x_torch , b_torch]


    str_output = "("*100

    k = 30
    l = []
    sum_axis = [0 , 1]
    for i in range(k):
        op_id = np.random.randint(0, 3) if i>10 else np.random.randint(0 , 2) 
        idx1, idx2 = np.random.randint(0, 2), np.random.randint(0, 2)  
        first_element_plus = bucket_1[idx1] if i == 0 else int_[f"var_{i-1}"]
        first_element_torch = bucket_2[idx1] if i == 0 else int_[f"var_torch{i-1}"]
        if i==0:
            l.append(idx1)


        first_part = ("x" if (idx1==1) else "y") if i==0 else ""
        second_part = "+"if op_id==1 else "*"
        third_part = "x)" if idx2==1 else "y)"

        str_output+=first_part +second_part+third_part

        l.append(idx2)
        if op_id == 1:  # Addition

            
            int_[f"var_{i}"] = first_element_plus + bucket_1[idx2] 
            int_[f"var_torch{i}"] = first_element_torch + bucket_2[idx2]
        elif op_id==0:  # Multiplication
            int_[f"var_{i}"] = first_element_plus * bucket_1[idx2] 
            int_[f"var_torch{i}"] = first_element_torch * bucket_2[idx2]
        if i >10:
            if op_id==2:
                random_array = np.random.uniform(-10 , 10 , size = (np.random.randint(3 , 8) , first_element_plus.shape[-1]))
                int_[f"var_{i}"] = first_element_plus @ random_array.T
                int_[f"var_torch{i}"] = first_element_torch @ torch.tensor(random_array).T

                random_array = np.random.uniform(-10 , 10 , size = (int_[f"var_{i}"].shape))
                torch_random_array = torch.tensor(random_array)
                bucket_1[0] , bucket_1[1] = random_array, random_array
                bucket_2[0] , bucket_2[1] = torch_random_array ,torch_random_array
        # summed = False
        # if summed:
        #     if np.random.randint(0 , 2)==1:
        #         summed = True
        #         what_axis_we_wanna_sum = np.random.randint(0 , len(sum_axis))
        #         keepdims =bool(np.random.randint(0 , 2))
                
        #         int_[f"var_{i}"] = int_[f"var_{i}"].sum(axis = sum_axis[what_axis_we_wanna_sum] , keepdims = keepdims)
        #         int_[f"var_torch{i}"] = int_[f"var_torch{i}"].sum(axis = sum_axis[what_axis_we_wanna_sum] , keepdims = keepdims)


        #         for i in range(2):
        #             bucket_1[i] = bucket_1[i].sum(sum_axis[what_axis_we_wanna_sum] , keepdims=keepdims)
        #             bucket_2[i] = bucket_2[i].sum(sum_axis[what_axis_we_wanna_sum] , keepdims=keepdims)

        #         sum_axis.pop(-1)







    final = int_[f"var_{k-1}"] .sum()
    final_torch= int_[f"var_torch{k-1}"].sum()

    if not requires_grad_b and not requires_grad_a:
        with pytest.raises(ValueError) as exc_info:
            final.backward()
        assert str(exc_info.value) == "root tensor does not require grad"
        return
        

    # x = x.sum()
    if all(l) and not requires_grad_b:
        with pytest.raises(ValueError) as exc_info:
            final.backward()

        assert str(exc_info.value) == "root tensor does not require grad"
        return
    if not any(l) and not requires_grad_a:
        with pytest.raises(ValueError) as exc_info:
            print("got here")
            final.backward()

        assert str(exc_info.value) == "root tensor does not require grad"
        return



    final.backward()
    final_torch.backward()

    grad_a = x_ref.grad
    grad_b = b_red.grad



    if x_ref.requires_grad:
        if x_torch_ref.grad is None:
            torch_grad_a = np.zeros_like(grad_a)
        else:
            torch_grad_a = x_torch_ref.grad.numpy()

        np.testing.assert_allclose(
            grad_a, torch_grad_a, rtol=0.10
        )
    if b_red.requires_grad:
        if b_torch_ref.grad is None:
            torch_grad_b = np.zeros_like(grad_b)
        else:
            torch_grad_b = b_torch_ref.grad.numpy()
        np.testing.assert_allclose(
            grad_b, torch_grad_b , rtol=0.10
        )
