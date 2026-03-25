
# Autograd

A mini homemade autograd engine built with NumPy.

This project is a simple automatic differentiation engine inspired by PyTorch’s `autograd`.  
It builds a computational graph, supports reverse-mode automatic differentiation, and computes gradients using the chain rule.

## What this project does

The goal of this project is to show how automatic differentiation works from scratch.

Instead of building large sparse Jacobian matrices, each operation has its own backward function that computes gradients locally. This makes the implementation cleaner and more efficient.

## Main ideas

- **Computational graph**: every operation is added to a graph so gradients can be tracked.
- **Reverse-mode autodiff**: gradients are computed backward from the final output.
- **Chain rule**: each node in the graph passes gradients to the previous nodes.
- **Vector-Jacobian products (VJPs)**: each operation implements its backward pass directly.
- **Efficient gradient computation**: avoids constructing large sparse matrices.

## Features

- Tensors with gradient tracking
- Custom `.backward()` implementation
- Reverse-mode automatic differentiation
- Chain rule based gradient propagation
- NumPy-based tensor operations
- Modular backward functions for each operation

## Supported operations

- Addition
- Subtraction
- Multiplication
- Matrix multiplication
- Mean
- Sum
- Transpose

## Project structure

- `tensor.py` — tensor class and autograd logic
- `context.py` — saves information needed for backward pass
- `base/` — base classes for tensors and functions
- `functions/` — operation-specific forward and backward implementations
- `utils.py` — helper functions

## Why this project is useful

This project helped me understand how deep learning libraries work internally.  
It was also a good exercise in graph theory, calculus, and efficient gradient computation.

## Example

```python
from tensor import Tensor

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
w = Tensor([2.0, 3.0, 4.0], requires_grad=True)

y = (x * w).sum()
y.backward()

print(x.grad)
print(w.grad)
```

## How it works

1. Create tensors
2. Apply operations
3. Build a computational graph
4. Call `.backward()` on the final output
5. Gradients flow backward through the graph

## Future work

- Add more operations
- Improve testing
- Add broadcasting support
- Add more examples and documentation

## Author

Mohammed Ahmed