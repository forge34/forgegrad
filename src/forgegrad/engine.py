"""Scalar autograd engine: the `Value` type and its backward pass."""

from __future__ import annotations
from math import cos, exp, log, sin


class Value:
    def __init__(self, data, op="", label="", children=()) -> None:
        self.data = data
        self.label = label
        self.children = tuple(children)
        self._op = op
        self._grad = 0
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, "+", "", [self, other])

        def _backward():
            self._grad += out._grad
            other._grad += out._grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, "-", "", [self, other])

        def _backward():
            self._grad += out._grad
            other._grad -= out._grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, "*", "", [self, other])

        def _backward():
            self._grad += other.data * out._grad
            other._grad += self.data * out._grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def exp(self):
        x = self.data
        out = Value(exp(x), children=[self])

        def _backward():
            self._grad += out._grad * out.data

        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data**other.data, "pow", "", [self, other])

        def _backward():
            self._grad += out._grad * other.data * self.data ** (other.data - 1)
            if self.data > 0:
                other._grad += out._grad * out.data * log(self.data)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        return self * (other if isinstance(other, Value) else Value(other)) ** -1

    def __rtruediv__(self, other):
        return (self ** -1) * other

    def __rpow__(self, other):
        return Value(other) ** self

    def sin(self):
        x = self.data
        out = Value(sin(x), label="sin", children=[self])

        def _backward():
            self._grad += out._grad * cos(x)

        out._backward = _backward

        return out

    def relU(self):
        out = Value(self.data if self.data > 0 else 0, "ReLU", "", [self])

        def _backward():
            if self.data > 0:
                self._grad += out._grad

        out._backward = _backward
        return out

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self._grad = 1
        for v in reversed(topo):
            v._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
