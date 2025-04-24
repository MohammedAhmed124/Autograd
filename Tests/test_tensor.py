import numpy as np
import torch
import pytest
from tensor import tensor

@pytest.mark.parametrize("case", range(10000))
def test_tensor_function(case):
    size = (np.random.randint(1, 20), np.random.randint(1, 20))

    a = tensor(np.random.randint(-10, 10, size=size), requires_grad=True)
    b = tensor(np.random.randint(-10, 10, size=size), requires_grad=True)

    a_torch = torch.tensor(a, requires_grad=True, dtype=torch.float32)
    b_torch = torch.tensor(b, requires_grad=True, dtype=torch.float32)

    a_first, a_first_torch = a, a_torch
    b_first, b_first_torch = b, b_torch

    # Randomly select operation (either addition or multiplication)
    bucket_1 = [a, b]
    bucket_2 = [a_torch, b_torch]
    int_ = {}
    k = 100  # number of operations
    for i in range(k):
        op_id = np.random.randint(0, 2)
        idx1, idx2 = np.random.randint(0, 2), np.random.randint(0, 2)

        int_[f"var_{i}"] = bucket_1[idx1] + bucket_1[idx2] if op_id == 1 else bucket_1[idx1] * bucket_1[idx2]
        int_[f"var_torch{i}"] = bucket_2[idx1] + bucket_2[idx2] if op_id == 1 else bucket_2[idx1] * bucket_2[idx2]

    result = int_[f"var_{k - 1}"].sum()
    result_torch = int_[f"var_torch{k - 1}"].sum()

    # Backpropagation for both
    result.backward()
    result_torch.backward()

    # Compare the gradients
    grad_a = np.array(a_first.grad) if isinstance(a_first.grad, tensor) else a_first.grad
    grad_b = np.array(b_first.grad) if isinstance(b_first.grad, tensor) else b_first.grad

    if a_first.requires_grad:
        if a_first_torch.grad is None:
            torch_grad_a = np.zeros_like(grad_a)
        else:
            torch_grad_a = a_first_torch.grad.numpy()

        np.testing.assert_equal(
            grad_a, torch_grad_a
        )
    if b_first.requires_grad:
        if b_first_torch.grad is None:
            torch_grad_b = np.zeros_like(grad_b)
        else:
            torch_grad_b = b_first_torch.grad.numpy()
        np.testing.assert_equal(
            grad_b, torch_grad_b
        )
