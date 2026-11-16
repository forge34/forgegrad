# forgegrad

A from-scratch implementation of scalar reverse-mode autodiff and a tiny neural
network library, built by following Andrej Karpathy's
[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0).

Inspired by [karpathy/micrograd](https://github.com/karpathy/micrograd) (MIT);
the code here is my own.

## Layout

```
src/forgegrad/
├── engine.py   # Value: scalar autograd node and backward pass
└── nn.py       # Neuron, Layer, MLP
```

## Setup

```bash
uv sync
```

## Usage

```python
import forgegrad
```
