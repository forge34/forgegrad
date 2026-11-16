"""Scalar autograd engine: the `Value` type and its backward pass."""

from __future__ import annotations


class Value:
    def __init__(self, data, op="", label="", children=()) -> None:
        self.data = data
        self.label = label
        self.children = tuple(children)
        self._op = op
        self._grad = 0
        self._backward = lambda: None

    def __add__(self, other: Value):

        out = Value(self.data + other.data, "+", "", [self, other])

        def _backward():
            self._grad += out._grad
            other._grad += out._grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, "*", "", [self, other])

        def _backward():
            self._grad += other.data * out._grad
            other._grad += self._grad * out._grad

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
